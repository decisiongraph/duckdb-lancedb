#define DUCKDB_EXTENSION_MAIN

#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "duckdb.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/execution/index/index_type_set.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	// Register LANCE index type
	IndexType lance_type;
	lance_type.name = LanceIndex::TYPE_NAME;
	lance_type.create_instance = LanceIndex::Create;
	lance_type.create_plan = LanceIndex::CreatePlan;
	db.config.GetIndexTypes().RegisterIndexType(lance_type);

	// Register table functions
	RegisterLanceSearchFunction(loader);
	RegisterLanceCreateAnnIndexFunction(loader);
	RegisterLanceInfoFunction(loader);

	// Register optimizer
	RegisterLanceOptimizer(db);
}

void LancedbExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string LancedbExtension::Name() {
	return "lancedb";
}

std::string LancedbExtension::Version() const {
#ifdef EXT_VERSION_LANCEDB
	return EXT_VERSION_LANCEDB;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(lancedb, loader) {
	duckdb::LoadInternal(loader);
}
}
