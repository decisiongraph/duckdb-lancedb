#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

// ========================================
// lance_search(table, index, query_vec, k)
// Returns (row_id BIGINT, distance FLOAT)
// ========================================

struct LanceSearchBindData : public TableFunctionData {
	string table_name;
	string index_name;
	vector<float> query;
	int32_t k;
};

struct LanceSearchState : public GlobalTableFunctionState {
	vector<row_t> row_ids;
	vector<float> distances;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> LanceSearchBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto bind_data = make_uniq<LanceSearchBindData>();
	bind_data->table_name = input.inputs[0].GetValue<string>();
	bind_data->index_name = input.inputs[1].GetValue<string>();

	auto list_val = input.inputs[2];
	auto &children = ListValue::GetChildren(list_val);
	for (auto &child : children) {
		bind_data->query.push_back(child.GetValue<float>());
	}

	bind_data->k = input.inputs[3].GetValue<int32_t>();

	return_types.push_back(LogicalType::BIGINT);
	return_types.push_back(LogicalType::FLOAT);
	names.push_back("row_id");
	names.push_back("distance");
	return std::move(bind_data);
}

static unique_ptr<GlobalTableFunctionState> LanceSearchInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<LanceSearchState>();
	auto &bind = input.bind_data->Cast<LanceSearchBindData>();

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
	auto results =
	    lance_idx.Search(bind.query.data(), static_cast<int32_t>(bind.query.size()), bind.k);

	for (auto &result : results) {
		state->row_ids.push_back(result.first);
		state->distances.push_back(result.second);
	}

	return std::move(state);
}

static void LanceSearchScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LanceSearchState>();

	if (state.position >= state.row_ids.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.row_ids.size() - state.position);

	auto rowid_data = FlatVector::GetData<int64_t>(output.data[0]);
	auto dist_data = FlatVector::GetData<float>(output.data[1]);

	for (idx_t i = 0; i < chunk_size; i++) {
		rowid_data[i] = state.row_ids[state.position + i];
		dist_data[i] = state.distances[state.position + i];
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

static unique_ptr<NodeStatistics> LanceSearchCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind = bind_data_p->Cast<LanceSearchBindData>();
	return make_uniq<NodeStatistics>(bind.k, bind.k);
}

void RegisterLanceSearchFunction(ExtensionLoader &loader) {
	TableFunction func(
	    "lance_search",
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::INTEGER},
	    LanceSearchScan, LanceSearchBind, LanceSearchInit);
	func.cardinality = LanceSearchCardinality;
	loader.RegisterFunction(func);
}

} // namespace duckdb
