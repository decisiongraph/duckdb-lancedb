#pragma once

#include "duckdb/execution/index/bound_index.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/execution/index/index_pointer.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/storage/data_table.hpp"
#include "rust_ffi.hpp"

#include <unordered_map>

namespace duckdb {

class DuckTableEntry;

// ========================================
// LanceIndex: BoundIndex for Lance vector search
// ========================================

class LanceIndex final : public BoundIndex {
public:
	static constexpr auto TYPE_NAME = "LANCE";

	LanceIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
	           TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	           AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	           const IndexStorageInfo &info = IndexStorageInfo());

	~LanceIndex() override;

	static unique_ptr<BoundIndex> Create(CreateIndexInput &input);
	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// BoundIndex interface
	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	void CommitDrop(IndexLock &lock) override;
	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;
	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;
	idx_t GetInMemorySize(IndexLock &state) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
	string VerifyAndToString(IndexLock &state, const bool only_verify) override;
	void VerifyAllocations(IndexLock &state) override;
	void VerifyBuffers(IndexLock &l) override;
	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override;

	// ANN search
	vector<pair<row_t, float>> Search(const float *query, int32_t dimension, int32_t k);

	// Build ANN index on the Lance dataset
	void CreateAnnIndex(int32_t num_partitions, int32_t num_sub_vectors);

	int32_t GetDimension() const {
		return dimension_;
	}
	const string &GetMetric() const {
		return metric_;
	}
	idx_t GetVectorCount() const {
		return rust_handle_ ? static_cast<idx_t>(LanceDetachedCount(rust_handle_)) : 0;
	}
	bool HasPendingDeletes() const {
		return has_pending_deletes_;
	}

	friend class PhysicalCreateLanceIndex;

private:
	void PersistToDisk();
	void LoadFromStorage(const IndexStorageInfo &info);
	string GetLancePath();

	// Rust Lance handle
	LanceHandle rust_handle_ = nullptr;

	// Cached Lance dataset path (generated once, reused)
	string lance_path_;

	// Parameters
	int32_t dimension_ = 0;
	string metric_ = "l2";
	int32_t nprobes_ = 20;
	int32_t refine_factor_ = 1;

	// Label <-> row_t mapping
	vector<row_t> label_to_rowid_;
	unordered_map<row_t, int64_t> rowid_to_label_;

	// Track whether deletes occurred since last vacuum
	bool has_pending_deletes_ = false;

	// Block storage (metadata only)
	unique_ptr<FixedSizeAllocator> block_allocator_;
	IndexPointer root_block_ptr_;
	bool is_dirty_ = false;
};

// ========================================
// PhysicalCreateLanceIndex
// ========================================

class PhysicalCreateLanceIndex : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CREATE_INDEX;

	PhysicalCreateLanceIndex(PhysicalPlan &physical_plan, LogicalOperator &op, TableCatalogEntry &table,
	                         const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
	                         vector<unique_ptr<Expression>> unbound_expressions, idx_t estimated_cardinality,
	                         unique_ptr<AlterTableInfo> alter_table_info = nullptr);

	DuckTableEntry &table;
	vector<column_t> storage_ids;
	unique_ptr<CreateIndexInfo> info;
	vector<unique_ptr<Expression>> unbound_expressions;
	unique_ptr<AlterTableInfo> alter_table_info;

public:
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
	bool IsSource() const override {
		return true;
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
};

} // namespace duckdb
