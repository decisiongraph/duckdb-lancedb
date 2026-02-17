#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LancedbExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

void RegisterLanceSearchFunction(ExtensionLoader &loader);
void RegisterLanceCreateAnnIndexFunction(ExtensionLoader &loader);
void RegisterLanceCreateHnswIndexFunction(ExtensionLoader &loader);
void RegisterLanceInfoFunction(ExtensionLoader &loader);
void RegisterLanceOptimizer(DatabaseInstance &db);

} // namespace duckdb
