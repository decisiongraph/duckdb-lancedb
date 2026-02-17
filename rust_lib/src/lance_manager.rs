//! LanceDB index manager: create, add vectors, search, manage Lance datasets.

use anyhow::{anyhow, Result};
use arrow_array::{
    Array, ArrayRef, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    FixedSizeListArray, StructArray,
};
use arrow_schema::{DataType, Field, Schema};
use arrow::compute::cast;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use futures_util::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table as LanceTable};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use crate::runtime;

/// Core LanceDB index handle.
pub struct LanceIndex {
    #[allow(dead_code)]
    connection: Connection,
    table: Option<LanceTable>,
    table_name: String,
    dimension: usize,
    metric: String,
    next_label: AtomicI64,
    schema: Arc<Schema>,
}

impl LanceIndex {
    /// Create a new Lance dataset at the given path (vector-only).
    pub fn create(db_path: &str, dimension: usize, metric: &str, table_name: &str) -> Result<Self> {
        let connection = runtime::block_on(lancedb::connect(db_path).execute())?;

        let schema = Self::build_vector_schema(dimension);
        let table_name = table_name.to_string();

        // Create empty table with schema (drop existing if present)
        let empty_batch = Self::empty_batch_from_schema(&schema)?;
        let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], schema.clone());
        let _ = runtime::block_on(connection.drop_table(&table_name));
        let table = runtime::block_on(
            connection
                .create_table(&table_name, Box::new(batches))
                .execute(),
        )?;

        Ok(Self {
            connection,
            table: Some(table),
            table_name,
            dimension,
            metric: metric.to_string(),
            next_label: AtomicI64::new(0),
            schema,
        })
    }

    /// Create a new Lance dataset from an Arrow schema (multi-column).
    ///
    /// The FFI schema describes the data columns (vector + extras).
    /// A label column is prepended automatically.
    ///
    /// # Safety
    /// Caller must pass a valid pointer to an Arrow C Data Interface ArrowSchema struct.
    pub unsafe fn create_from_arrow(
        db_path: &str,
        ffi_schema_ptr: *mut FFI_ArrowSchema,
        metric: &str,
        table_name: &str,
    ) -> Result<Self> {
        // Import schema from FFI (borrows, does not consume)
        let ffi_schema = &*ffi_schema_ptr;
        let imported_schema = Schema::try_from(ffi_schema)
            .map_err(|e| anyhow!("FFI schema import failed: {}", e))?;

        // Find vector dimension from FixedSizeList column
        let mut dimension = 0usize;
        for field in imported_schema.fields() {
            if let DataType::FixedSizeList(_, dim) = field.data_type() {
                dimension = *dim as usize;
                break;
            }
        }
        if dimension == 0 {
            return Err(anyhow!("no FixedSizeList column found in schema"));
        }

        // Build table schema: prepend label column, then imported fields
        let mut table_fields: Vec<Arc<Field>> = vec![Arc::new(Field::new("label", DataType::Int64, false))];
        for field in imported_schema.fields() {
            // Rename vector column's child field to "item" (DuckDB uses "")
            if let DataType::FixedSizeList(_, dim) = field.data_type() {
                let fixed_field = Field::new(
                    field.name(),
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        *dim,
                    ),
                    field.is_nullable(),
                );
                table_fields.push(Arc::new(fixed_field));
            } else {
                table_fields.push(Arc::new(field.as_ref().clone()));
            }
        }
        let table_schema = Arc::new(Schema::new(table_fields));

        // Create empty batch
        let empty_batch = Self::empty_batch_from_schema(&table_schema)?;

        // Create table
        let connection = runtime::block_on(lancedb::connect(db_path).execute())?;
        let table_name = table_name.to_string();
        let _ = runtime::block_on(connection.drop_table(&table_name));
        let batches = RecordBatchIterator::new(vec![Ok(empty_batch)], table_schema.clone());
        let table = runtime::block_on(
            connection
                .create_table(&table_name, Box::new(batches))
                .execute(),
        )?;

        Ok(Self {
            connection,
            table: Some(table),
            table_name,
            dimension,
            metric: metric.to_string(),
            next_label: AtomicI64::new(0),
            schema: table_schema,
        })
    }

    /// Reopen an existing Lance dataset, deriving schema from the table.
    pub fn open(db_path: &str, table_name: &str, metric: &str) -> Result<Self> {
        let connection = runtime::block_on(lancedb::connect(db_path).execute())?;
        let table_name_str = table_name.to_string();
        let table = runtime::block_on(connection.open_table(&table_name_str).execute())?;

        // Derive schema from the Lance table
        let table_schema = Self::read_table_schema(&table)?;

        // Derive dimension from the vector column
        let mut dimension = 0usize;
        for field in table_schema.fields() {
            if let DataType::FixedSizeList(_, dim) = field.data_type() {
                dimension = *dim as usize;
                break;
            }
        }
        if dimension == 0 {
            // Fallback: might be a vector-only table with known schema
            return Err(anyhow!("cannot determine dimension from table schema"));
        }

        // Use MAX(label)+1, not count_rows() — count is wrong after deletes.
        let next_label = Self::query_max_label(&table)? + 1;

        Ok(Self {
            connection,
            table: Some(table),
            table_name: table_name_str,
            dimension,
            metric: metric.to_string(),
            next_label: AtomicI64::new(next_label),
            schema: table_schema,
        })
    }

    /// Read the schema from a Lance table via its metadata (no data query needed).
    fn read_table_schema(table: &LanceTable) -> Result<Arc<Schema>> {
        Ok(runtime::block_on(table.schema())?)
    }

    /// Whether this index has extra columns beyond label + vector.
    pub fn has_extra_columns(&self) -> bool {
        self.schema.fields().len() > 2
    }

    /// Get the table name.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Get the dimension of vectors in this index.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the metric name.
    pub fn metric(&self) -> &str {
        &self.metric
    }

    /// Clone the table handle. LanceTable is Arc-based (O(1) clone).
    fn get_table(&self) -> Result<LanceTable> {
        self.table
            .as_ref()
            .ok_or_else(|| anyhow!("table not open"))
            .cloned()
    }

    /// Add a single vector. Returns the assigned label.
    pub fn add_vector(&self, vector: &[f32]) -> Result<i64> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "expected dimension {}, got {}",
                self.dimension,
                vector.len()
            ));
        }

        let label = self.next_label.fetch_add(1, Ordering::Relaxed);
        let batch = self.make_batch(&[label], &[vector])?;

        let table = self.get_table()?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
        runtime::block_on(table.add(Box::new(batches)).execute())?;

        Ok(label)
    }

    /// Add a batch of contiguous vectors. Returns labels.
    ///
    /// `vectors` must be a flat contiguous array: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...].
    pub fn add_batch(&self, vectors: &[f32], num_vectors: usize) -> Result<Vec<i64>> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(anyhow!("vector data size mismatch"));
        }

        let start_label = self.next_label.fetch_add(num_vectors as i64, Ordering::Relaxed);
        let labels: Vec<i64> = (start_label..start_label + num_vectors as i64).collect();

        let batch = self.make_batch_contiguous(&labels, vectors)?;

        let table = self.get_table()?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
        runtime::block_on(table.add(Box::new(batches)).execute())?;

        Ok(labels)
    }

    /// Add a batch of rows via Arrow C Data Interface (multi-column path).
    ///
    /// The incoming Arrow struct has columns matching the table schema minus the label column.
    /// Labels are auto-generated. Returns assigned labels.
    ///
    /// # Safety
    /// Caller must pass valid pointers to Arrow C Data Interface structs.
    pub unsafe fn add_batch_arrow(
        &self,
        ffi_schema_ptr: *mut FFI_ArrowSchema,
        ffi_array_ptr: *mut FFI_ArrowArray,
    ) -> Result<Vec<i64>> {
        // Take ownership of the ArrowArray, leaving an empty one in C++ to prevent double-free
        let ffi_array = std::mem::replace(&mut *ffi_array_ptr, FFI_ArrowArray::empty());
        // Reference the schema (C++ still owns it and will release)
        let ffi_schema_ref = &*ffi_schema_ptr;

        // Import to Arrow arrays
        let array_data = arrow::ffi::from_ffi(ffi_array, ffi_schema_ref)
            .map_err(|e| anyhow!("Arrow FFI import failed: {}", e))?;
        let struct_array = StructArray::from(array_data);
        let num_rows = struct_array.len();

        if num_rows == 0 {
            return Ok(vec![]);
        }

        // Generate labels
        let start_label = self.next_label.fetch_add(num_rows as i64, Ordering::Relaxed);
        let labels: Vec<i64> = (start_label..start_label + num_rows as i64).collect();
        let label_array = Int64Array::from(labels.clone());

        // Build columns: [label, vector, extra1, extra2, ...]
        // Use cast to match the table schema types (e.g., FixedSizeList child field name may differ)
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + struct_array.num_columns());
        columns.push(Arc::new(label_array));
        for i in 0..struct_array.num_columns() {
            let col = struct_array.column(i);
            let target_type = self.schema.field(i + 1).data_type();
            if col.data_type() == target_type {
                columns.push(col.clone());
            } else {
                // Cast to match schema (handles FixedSizeList child field name differences)
                let casted = cast(col.as_ref(), target_type)
                    .map_err(|e| anyhow!("cast column {} failed: {}", i, e))?;
                columns.push(casted);
            }
        }

        let batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| anyhow!("RecordBatch schema mismatch: {}", e))?;

        let table = self.get_table()?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], self.schema.clone());
        runtime::block_on(table.add(Box::new(batches)).execute())?;

        Ok(labels)
    }

    /// Merge live rows from source into self. All done in Rust, no extra FFI round-trip.
    ///
    /// `live_source_labels` are labels in the source that should be copied (not tombstoned).
    /// Returns Vec<(old_label, new_label)> for the caller to update its mappings.
    pub fn merge_from(
        &self,
        source: &LanceIndex,
        live_source_labels: &[i64],
    ) -> Result<Vec<(i64, i64)>> {
        if live_source_labels.is_empty() {
            return Ok(vec![]);
        }

        let source_table = source.get_table()?;

        // Build a predicate to select only the live labels
        let csv: String = live_source_labels
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let predicate = format!("label IN ({})", csv);

        let results = runtime::block_on(
            source_table
                .query()
                .only_if(predicate)
                .execute(),
        )?;

        // Collect all batches from source
        let source_batches: Vec<RecordBatch> = runtime::block_on(async {
            let mut batches = Vec::new();
            let mut stream = results;
            while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| anyhow!("stream error: {}", e))?
            {
                batches.push(batch);
            }
            Ok::<_, anyhow::Error>(batches)
        })?;

        let mut label_mapping = Vec::new();
        let table = self.get_table()?;

        for batch in &source_batches {
            if batch.num_rows() == 0 {
                continue;
            }

            // Extract old labels
            let old_label_col = batch
                .column_by_name("label")
                .ok_or_else(|| anyhow!("missing label column in source"))?;
            let old_labels = old_label_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("label column not Int64"))?;

            // Generate new labels
            let num_rows = batch.num_rows();
            let start_label = self.next_label.fetch_add(num_rows as i64, Ordering::Relaxed);
            let new_labels: Vec<i64> = (start_label..start_label + num_rows as i64).collect();
            let new_label_array = Int64Array::from(new_labels.clone());

            // Record old→new mapping
            for i in 0..num_rows {
                label_mapping.push((old_labels.value(i), new_labels[i]));
            }

            // Build new batch: replace label column with new labels, keep everything else
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
            columns.push(Arc::new(new_label_array));
            for i in 1..batch.num_columns() {
                columns.push(batch.column(i).clone());
            }

            let new_batch = RecordBatch::try_new(self.schema.clone(), columns)
                .map_err(|e| anyhow!("merge batch schema mismatch: {}", e))?;

            let batches_iter =
                RecordBatchIterator::new(vec![Ok(new_batch)], self.schema.clone());
            runtime::block_on(table.add(Box::new(batches_iter)).execute())?;
        }

        Ok(label_mapping)
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobes: usize,
        refine_factor: usize,
    ) -> Result<Vec<(i64, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "expected query dimension {}, got {}",
                self.dimension,
                query.len()
            ));
        }

        let table = self.get_table()?;

        let results = runtime::block_on(
            table
                .vector_search(query)
                .map_err(|e| anyhow!("search setup: {}", e))?
                .limit(k)
                .nprobes(nprobes)
                .refine_factor(refine_factor as u32)
                .execute(),
        )?;

        let mut output = Vec::with_capacity(k);

        runtime::block_on(async {
            let mut stream = results;
            while let Some(batch) = stream.try_next().await
                .map_err(|e| anyhow!("stream error: {}", e))? {
                let label_col = batch
                    .column_by_name("label")
                    .ok_or_else(|| anyhow!("missing label column"))?;
                let labels = label_col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("label column not Int64"))?;

                let dist_col = batch
                    .column_by_name("_distance")
                    .ok_or_else(|| anyhow!("missing _distance column"))?;
                let distances = dist_col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("distance column not Float32"))?;

                for i in 0..batch.num_rows() {
                    output.push((labels.value(i), distances.value(i)));
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;

        Ok(output)
    }

    /// Delete a vector by label.
    pub fn delete(&self, label: i64) -> Result<()> {
        let table = self.get_table()?;
        runtime::block_on(table.delete(&format!("label = {}", label)))?;
        Ok(())
    }

    /// Delete multiple vectors by label in a single operation.
    pub fn delete_batch(&self, labels: &[i64]) -> Result<()> {
        if labels.is_empty() {
            return Ok(());
        }
        let table = self.get_table()?;

        let csv: String = labels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
        let predicate = format!("label IN ({})", csv);
        runtime::block_on(table.delete(&predicate))?;
        Ok(())
    }

    /// Count vectors.
    pub fn count(&self) -> Result<u64> {
        let table = self.get_table()?;
        let count = runtime::block_on(table.count_rows(None))?;
        Ok(count as u64)
    }

    /// Create an ANN index (IVF_PQ).
    ///
    /// Pass 0 for num_partitions or num_sub_vectors to use LanceDB defaults.
    pub fn create_ann_index(
        &self,
        num_partitions: u32,
        num_sub_vectors: u32,
    ) -> Result<()> {
        let table = self.get_table()?;

        use lancedb::index::vector::IvfPqIndexBuilder;
        use lancedb::index::Index;

        let distance_type = match self.metric.as_str() {
            "cosine" => lancedb::DistanceType::Cosine,
            "dot" | "ip" => lancedb::DistanceType::Dot,
            _ => lancedb::DistanceType::L2,
        };

        let mut builder = IvfPqIndexBuilder::default().distance_type(distance_type);
        if num_partitions > 0 {
            builder = builder.num_partitions(num_partitions);
        }
        if num_sub_vectors > 0 {
            builder = builder.num_sub_vectors(num_sub_vectors);
        }

        runtime::block_on(
            table
                .create_index(&["vector"], Index::IvfPq(builder))
                .replace(true)
                .execute(),
        )?;

        Ok(())
    }

    /// Create an ANN index (IVF_HNSW_SQ).
    ///
    /// `m` is the number of edges per node in the HNSW graph.
    /// `ef_construction` is the search width during index build.
    pub fn create_hnsw_index(
        &self,
        m: u32,
        ef_construction: u32,
    ) -> Result<()> {
        let table = self.get_table()?;

        use lancedb::index::vector::IvfHnswSqIndexBuilder;
        use lancedb::index::Index;

        let distance_type = match self.metric.as_str() {
            "cosine" => lancedb::DistanceType::Cosine,
            "dot" | "ip" => lancedb::DistanceType::Dot,
            _ => lancedb::DistanceType::L2,
        };

        let mut builder = IvfHnswSqIndexBuilder::default()
            .distance_type(distance_type);
        if m > 0 {
            builder = builder.num_edges(m);
        }
        if ef_construction > 0 {
            builder = builder.ef_construction(ef_construction);
        }

        runtime::block_on(
            table
                .create_index(&["vector"], Index::IvfHnswSq(builder))
                .replace(true)
                .execute(),
        )?;

        Ok(())
    }

    /// Compact the dataset (optimize storage).
    pub fn compact(&self) -> Result<()> {
        let table = self.get_table()?;
        runtime::block_on(table.optimize(lancedb::table::OptimizeAction::All)).map(|_| ())?;
        Ok(())
    }

    /// Get a vector by label.
    pub fn get_vector(&self, label: i64) -> Result<Vec<f32>> {
        let table = self.get_table()?;

        let results = runtime::block_on(
            table
                .query()
                .only_if(format!("label = {}", label))
                .execute(),
        )?;

        let batches: Vec<RecordBatch> = runtime::block_on(async {
            let mut batches = Vec::new();
            let mut stream = results;
            loop {
                match stream.try_next().await {
                    Ok(Some(batch)) => batches.push(batch),
                    Ok(None) => break,
                    Err(e) => return Err(anyhow!("stream error: {}", e)),
                }
            }
            Ok(batches)
        })?;

        for batch in &batches {
            if batch.num_rows() > 0 {
                let vec_col = batch
                    .column_by_name("vector")
                    .ok_or_else(|| anyhow!("missing vector column"))?;
                let list_array = vec_col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| anyhow!("vector not FixedSizeList"))?;
                let values = list_array
                    .value(0);
                let float_array = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| anyhow!("vector values not Float32"))?;
                return Ok(float_array.values().to_vec());
            }
        }

        Err(anyhow!("label {} not found", label))
    }

    /// Get all vectors as a flat contiguous f32 array and their labels.
    /// Returns (labels, flat_vectors) where flat_vectors has length labels.len() * dimension.
    pub fn get_all_vectors(&self) -> Result<(Vec<i64>, Vec<f32>)> {
        let table = self.get_table()?;

        let results = runtime::block_on(
            table
                .query()
                .execute(),
        )?;

        let mut all_labels = Vec::new();
        let mut all_vectors = Vec::new();

        runtime::block_on(async {
            let mut stream = results;
            while let Some(batch) = stream.try_next().await
                .map_err(|e| anyhow!("stream error: {}", e))? {
                let label_col = batch
                    .column_by_name("label")
                    .ok_or_else(|| anyhow!("missing label column"))?;
                let labels = label_col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("label column not Int64"))?;

                let vec_col = batch
                    .column_by_name("vector")
                    .ok_or_else(|| anyhow!("missing vector column"))?;
                let list_array = vec_col
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| anyhow!("vector not FixedSizeList"))?;

                for i in 0..batch.num_rows() {
                    all_labels.push(labels.value(i));
                    let values = list_array.value(i);
                    let float_array = values
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| anyhow!("vector values not Float32"))?;
                    all_vectors.extend_from_slice(float_array.values());
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;

        Ok((all_labels, all_vectors))
    }

    // Internal helpers

    /// Query MAX(label) from the table. Returns -1 if empty.
    fn query_max_label(table: &LanceTable) -> Result<i64> {
        let count = runtime::block_on(table.count_rows(None))?;
        if count == 0 {
            return Ok(-1);
        }

        let results = runtime::block_on(table.query().execute())?;

        let mut max_label: i64 = -1;
        runtime::block_on(async {
            let mut stream = results;
            while let Some(batch) = stream
                .try_next()
                .await
                .map_err(|e| anyhow!("stream error: {}", e))?
            {
                let label_col = batch
                    .column_by_name("label")
                    .ok_or_else(|| anyhow!("missing label column"))?;
                let labels = label_col
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow!("label column not Int64"))?;
                for i in 0..labels.len() {
                    let val = labels.value(i);
                    if val > max_label {
                        max_label = val;
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;

        Ok(max_label)
    }

    /// Build schema for vector-only tables (label + vector).
    fn build_vector_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("label", DataType::Int64, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
        ]))
    }

    /// Create an empty RecordBatch from a schema by generating empty arrays for each field.
    fn empty_batch_from_schema(schema: &Arc<Schema>) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|field| Self::empty_array_for_datatype(field.data_type()))
            .collect();
        Ok(RecordBatch::try_new(schema.clone(), columns)?)
    }

    /// Create an empty array for any Arrow DataType.
    fn empty_array_for_datatype(dt: &DataType) -> ArrayRef {
        use arrow_array::{BooleanArray, Int32Array, Float64Array, StringArray};
        match dt {
            DataType::Int64 => Arc::new(Int64Array::from(Vec::<i64>::new())),
            DataType::Int32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
            DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())),
            DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
            DataType::Utf8 => Arc::new(StringArray::from(Vec::<&str>::new())),
            DataType::Boolean => Arc::new(BooleanArray::from(Vec::<bool>::new())),
            DataType::FixedSizeList(child_field, dim) => {
                let values = Float32Array::from(Vec::<f32>::new());
                let list = FixedSizeListArray::new(child_field.clone(), *dim, Arc::new(values), None);
                Arc::new(list)
            }
            _ => Arc::new(StringArray::from(Vec::<&str>::new())), // fallback
        }
    }

    fn make_fixed_size_list(values: Float32Array, dimension: i32) -> FixedSizeListArray {
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let values_ref: ArrayRef = Arc::new(values);
        FixedSizeListArray::new(field, dimension, values_ref, None)
    }

    /// Build a RecordBatch from already-contiguous flat vector data (no re-flattening needed).
    fn make_batch_contiguous(&self, labels: &[i64], flat_vectors: &[f32]) -> Result<RecordBatch> {
        let label_array = Int64Array::from(labels.to_vec());
        let values = Float32Array::from(flat_vectors.to_vec());
        let list = Self::make_fixed_size_list(values, self.dimension as i32);
        Ok(RecordBatch::try_new(self.schema.clone(), vec![
            Arc::new(label_array),
            Arc::new(list),
        ])?)
    }

    fn make_batch(&self, labels: &[i64], vectors: &[&[f32]]) -> Result<RecordBatch> {
        let label_array = Int64Array::from(labels.to_vec());
        let flat_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let values = Float32Array::from(flat_values);
        let list = Self::make_fixed_size_list(values, self.dimension as i32);
        Ok(RecordBatch::try_new(self.schema.clone(), vec![
            Arc::new(label_array),
            Arc::new(list),
        ])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    #[test]
    fn test_next_label_unique_after_deletes() {
        let dir = temp_dir();
        let db_path = dir.path().join("test.lance");
        let db_path_str = db_path.to_str().unwrap();

        // Insert 5 vectors with labels [0,1,2,3,4]
        let idx = LanceIndex::create(db_path_str, 3, "l2", "vectors").unwrap();
        for i in 0..5 {
            let label = idx.add_vector(&[i as f32, 0.0, 0.0]).unwrap();
            assert_eq!(label, i);
        }

        // Delete labels [1,2]
        idx.delete(1).unwrap();
        idx.delete(2).unwrap();

        // Reopen — next insert must NOT produce label 3 (already exists)
        drop(idx);
        let idx2 = LanceIndex::open(db_path_str, "vectors", "l2").unwrap();
        let new_label = idx2.add_vector(&[99.0, 0.0, 0.0]).unwrap();
        assert!(
            new_label >= 5,
            "expected label >= 5 after deletes, got {new_label}"
        );
    }

    #[test]
    fn test_next_label_correct_on_empty_reopen() {
        let dir = temp_dir();
        let db_path = dir.path().join("test_empty.lance");
        let db_path_str = db_path.to_str().unwrap();

        let idx = LanceIndex::create(db_path_str, 2, "l2", "vectors").unwrap();
        drop(idx);

        let idx2 = LanceIndex::open(db_path_str, "vectors", "l2").unwrap();
        let label = idx2.add_vector(&[1.0, 2.0]).unwrap();
        assert_eq!(label, 0);
    }

    #[test]
    fn test_open_derives_schema() {
        let dir = temp_dir();
        let db_path = dir.path().join("test_schema.lance");
        let db_path_str = db_path.to_str().unwrap();

        let idx = LanceIndex::create(db_path_str, 3, "l2", "vectors").unwrap();
        for i in 0..5 {
            idx.add_vector(&[i as f32, 0.0, 0.0]).unwrap();
        }
        idx.delete(1).unwrap();
        idx.delete(2).unwrap();

        drop(idx);

        let idx2 = LanceIndex::open(db_path_str, "vectors", "l2").unwrap();
        let new_label = idx2.add_vector(&[99.0, 0.0, 0.0]).unwrap();
        assert!(
            new_label >= 5,
            "expected label >= 5 via open, got {new_label}"
        );
    }

    #[test]
    fn test_custom_table_name() {
        let dir = temp_dir();
        let db_path = dir.path().join("test_tbl.lance");
        let db_path_str = db_path.to_str().unwrap();

        // Two indexes in same dataset with different table names
        let idx_a = LanceIndex::create(db_path_str, 2, "l2", "idx_a").unwrap();
        let idx_b = LanceIndex::create(db_path_str, 2, "l2", "idx_b").unwrap();

        idx_a.add_vector(&[1.0, 0.0]).unwrap();
        idx_a.add_vector(&[2.0, 0.0]).unwrap();
        idx_b.add_vector(&[10.0, 0.0]).unwrap();

        assert_eq!(idx_a.count().unwrap(), 2);
        assert_eq!(idx_b.count().unwrap(), 1);

        // Reopen each — they stay independent
        drop(idx_a);
        drop(idx_b);
        let idx_a2 = LanceIndex::open(db_path_str, "idx_a", "l2").unwrap();
        let idx_b2 = LanceIndex::open(db_path_str, "idx_b", "l2").unwrap();
        assert_eq!(idx_a2.count().unwrap(), 2);
        assert_eq!(idx_b2.count().unwrap(), 1);
    }
}
