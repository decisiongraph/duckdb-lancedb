#include "lancedb_extension.hpp"
#include "lance_index.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/index_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/storage/data_table.hpp"

namespace duckdb {

struct LanceInfoEntry {
	string name;
	string table_name;
	string metric;
	int32_t dimension;
	int64_t vector_count;
};

struct LanceInfoState : public GlobalTableFunctionState {
	vector<LanceInfoEntry> entries;
	idx_t position = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> LanceInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("name");
	names.push_back("table_name");
	names.push_back("metric");
	names.push_back("dimension");
	names.push_back("vector_count");
	return make_uniq<TableFunctionData>();
}

static unique_ptr<GlobalTableFunctionState> LanceInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	auto state = make_uniq<LanceInfoState>();

	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type != "LANCE") {
				return;
			}

			LanceInfoEntry e;
			e.name = index_entry.name;
			e.table_name = index_entry.GetTableName();

			try {
				auto &catalog = Catalog::GetCatalog(context, "");
				auto &table_entry =
				    catalog.GetEntry<TableCatalogEntry>(context, DEFAULT_SCHEMA, e.table_name);
				auto &duck_table = table_entry.Cast<DuckTableEntry>();
				auto &storage = duck_table.GetStorage();
				auto &table_info = *storage.GetDataTableInfo();
				auto &indexes = table_info.GetIndexes();

				indexes.Bind(context, table_info, LanceIndex::TYPE_NAME);
				auto idx_ptr = indexes.Find(e.name);
				if (idx_ptr) {
					auto &lance_idx = idx_ptr->Cast<LanceIndex>();
					e.dimension = lance_idx.GetDimension();
					e.vector_count = static_cast<int64_t>(lance_idx.GetVectorCount());
					e.metric = lance_idx.GetMetric();
				}
			} catch (const std::exception &ex) {
				e.dimension = -1;
				e.vector_count = -1;
				e.metric = ex.what();
			}

			state->entries.push_back(std::move(e));
		});
	}

	return std::move(state);
}

static void LanceInfoScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LanceInfoState>();

	if (state.position >= state.entries.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t chunk_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.entries.size() - state.position);

	for (idx_t i = 0; i < chunk_size; i++) {
		auto &entry = state.entries[state.position + i];
		output.SetValue(0, i, Value(entry.name));
		output.SetValue(1, i, Value(entry.table_name));
		output.SetValue(2, i, Value(entry.metric));
		output.SetValue(3, i, Value::INTEGER(entry.dimension));
		output.SetValue(4, i, Value::BIGINT(entry.vector_count));
	}

	state.position += chunk_size;
	output.SetCardinality(chunk_size);
}

void RegisterLanceInfoFunction(ExtensionLoader &loader) {
	TableFunction func("lance_info", {}, LanceInfoScan, LanceInfoBind, LanceInfoInit);
	loader.RegisterFunction(func);
}

} // namespace duckdb
