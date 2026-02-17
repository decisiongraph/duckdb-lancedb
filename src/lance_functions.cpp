#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

// ========================================
// lance_create_ann_index(table, index, num_partitions, num_sub_vectors)
// Build IVF_PQ index for large datasets.
// ========================================

struct LanceCreateAnnBindData : public TableFunctionData {
	string table_name;
	string index_name;
	int32_t num_partitions;
	int32_t num_sub_vectors;
};

struct LanceCreateAnnState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> LanceCreateAnnBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<LanceCreateAnnBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();
	bind_data->num_partitions = input.inputs[2].GetValue<int32_t>();
	bind_data->num_sub_vectors = input.inputs[3].GetValue<int32_t>();

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("status");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> LanceCreateAnnInit(ClientContext &context,
                                                               TableFunctionInitInput &input) {
	return make_uniq<LanceCreateAnnState>();
}

static void LanceCreateAnnScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<LanceCreateAnnBindData>();
	auto &state = data.global_state->Cast<LanceCreateAnnState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &catalog = Catalog::GetCatalog(context, "");
	auto &table_entry = catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name);
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();

	indexes.Bind(context, table_info, LanceIndex::TYPE_NAME);

	auto index_ptr = indexes.Find(bind.index_name);
	if (!index_ptr) {
		throw InvalidInputException("Index '%s' not found on table '%s'", bind.index_name, bind.table_name);
	}

	auto &lance_idx = index_ptr->Cast<LanceIndex>();
	lance_idx.CreateAnnIndex(bind.num_partitions, bind.num_sub_vectors);

	output.data[0].SetValue(0, Value("ANN index created"));
	output.SetCardinality(1);
}

void RegisterLanceCreateAnnIndexFunction(ExtensionLoader &loader) {
	TableFunction func("lance_create_ann_index",
	                   {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
	                   LanceCreateAnnScan, LanceCreateAnnBind, LanceCreateAnnInit);
	loader.RegisterFunction(func);
}

// ========================================
// lance_create_hnsw_index(table, index, m, ef_construction)
// Build IVF_HNSW_SQ index for better recall.
// ========================================

struct LanceCreateHnswBindData : public TableFunctionData {
	string table_name;
	string index_name;
	int32_t m;
	int32_t ef_construction;
};

struct LanceCreateHnswState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> LanceCreateHnswBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<LanceCreateHnswBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();
	bind_data->m = input.inputs[2].GetValue<int32_t>();
	bind_data->ef_construction = input.inputs[3].GetValue<int32_t>();

	return_types.push_back(LogicalType::VARCHAR);
	names.push_back("status");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> LanceCreateHnswInit(ClientContext &context,
                                                                 TableFunctionInitInput &input) {
	return make_uniq<LanceCreateHnswState>();
}

static void LanceCreateHnswScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<LanceCreateHnswBindData>();
	auto &state = data.global_state->Cast<LanceCreateHnswState>();

	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &catalog = Catalog::GetCatalog(context, "");
	auto &table_entry = catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, bind.table_name);
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();

	indexes.Bind(context, table_info, LanceIndex::TYPE_NAME);

	auto index_ptr = indexes.Find(bind.index_name);
	if (!index_ptr) {
		throw InvalidInputException("Index '%s' not found on table '%s'", bind.index_name, bind.table_name);
	}

	auto &lance_idx = index_ptr->Cast<LanceIndex>();
	lance_idx.CreateHnswIndex(bind.m, bind.ef_construction);

	output.data[0].SetValue(0, Value("HNSW index created"));
	output.SetCardinality(1);
}

void RegisterLanceCreateHnswIndexFunction(ExtensionLoader &loader) {
	TableFunction func("lance_create_hnsw_index",
	                   {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::INTEGER},
	                   LanceCreateHnswScan, LanceCreateHnswBind, LanceCreateHnswInit);
	loader.RegisterFunction(func);
}

} // namespace duckdb
