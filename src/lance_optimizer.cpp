// ANN index scan optimizer for LANCE indexes.
// Rewrites: ORDER BY array_distance(col, query) LIMIT k → Lance index scan.
// Adapted from duckdb-annsearch/src/ann_optimizer.cpp.

#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

namespace duckdb {

// ========================================
// LanceIndexScan: replacement table function
// ========================================

struct LanceIndexScanBindData : public TableFunctionData {
	DuckTableEntry *table_entry = nullptr;
	string index_name;

	unsafe_unique_array<float> query_vector;
	idx_t vector_size = 0;
	idx_t limit = 100;

	vector<StorageIndex> storage_ids;
};

struct LanceIndexScanGlobalState : public GlobalTableFunctionState {
	vector<pair<row_t, float>> results;
	idx_t offset = 0;

	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<FunctionData> LanceIndexScanBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &,
                                                   vector<string> &) {
	throw InternalException("LanceIndexScan bind should not be called directly — set by optimizer");
}

static unique_ptr<GlobalTableFunctionState> LanceIndexScanInit(ClientContext &context,
                                                               TableFunctionInitInput &input) {
	auto state = make_uniq<LanceIndexScanGlobalState>();
	auto &bind_data = input.bind_data->Cast<LanceIndexScanBindData>();

	auto &storage = bind_data.table_entry->GetStorage();
	auto &table_info = *storage.GetDataTableInfo();
	auto &indexes = table_info.GetIndexes();

	indexes.Bind(context, table_info, LanceIndex::TYPE_NAME);
	auto idx_ptr = indexes.Find(bind_data.index_name);
	if (idx_ptr) {
		auto &lance_idx = idx_ptr->Cast<LanceIndex>();
		state->results = lance_idx.Search(bind_data.query_vector.get(),
		                                  static_cast<int32_t>(bind_data.vector_size),
		                                  static_cast<int32_t>(bind_data.limit));
	}

	return std::move(state);
}

static void LanceIndexScanScan(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind_data = data.bind_data->Cast<LanceIndexScanBindData>();
	auto &state = data.global_state->Cast<LanceIndexScanGlobalState>();

	auto remaining = state.results.size() - state.offset;
	if (remaining == 0) {
		output.SetCardinality(0);
		return;
	}

	auto batch_size = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);

	Vector row_ids_vec(LogicalType::ROW_TYPE, batch_size);
	auto row_ids_data = FlatVector::GetData<row_t>(row_ids_vec);
	for (idx_t i = 0; i < batch_size; i++) {
		row_ids_data[i] = state.results[state.offset + i].first;
	}

	auto &storage = bind_data.table_entry->GetStorage();
	auto &transaction = DuckTransaction::Get(context, storage.db);
	ColumnFetchState fetch_state;
	storage.Fetch(transaction, output, bind_data.storage_ids, row_ids_vec, batch_size, fetch_state);

	state.offset += batch_size;
	output.SetCardinality(batch_size);
}

// ========================================
// Optimizer: detect ORDER BY array_distance(...) LIMIT k
// ========================================

static bool IsArrayDistanceFunction(const Expression &expr) {
	if (expr.type != ExpressionType::BOUND_FUNCTION) {
		return false;
	}
	auto &func_expr = expr.Cast<BoundFunctionExpression>();
	return func_expr.function.name == "array_distance" || func_expr.function.name == "array_cosine_distance" ||
	       func_expr.function.name == "array_inner_product";
}

class LanceOptimizerExtension : public OptimizerExtension {
public:
	LanceOptimizerExtension() {
		optimize_function = Optimize;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		// Look for: LIMIT -> ORDER -> ... -> GET pattern
		TryRewrite(input, plan);
	}

private:
	static void TryRewrite(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &op) {
		// Recurse into children first
		for (auto &child : op->children) {
			TryRewrite(input, child);
		}

		// Match: LogicalLimit -> LogicalOrder -> ... -> LogicalGet
		if (op->type != LogicalOperatorType::LOGICAL_LIMIT) {
			return;
		}
		auto &limit_op = op->Cast<LogicalLimit>();

		if (op->children.empty() || op->children[0]->type != LogicalOperatorType::LOGICAL_ORDER) {
			return;
		}
		auto &order_op = op->children[0]->Cast<LogicalOrder>();

		if (order_op.orders.size() != 1) {
			return;
		}
		auto &order_expr = order_op.orders[0].expression;

		if (!IsArrayDistanceFunction(*order_expr)) {
			return;
		}

		// We found the pattern but full rewrite requires resolving the index.
		// For now, leave as-is — the lance_search() function is the recommended API.
		// Full optimizer rewrite can be added once the basic extension works.
	}
};

void RegisterLanceOptimizer(DatabaseInstance &db) {
	db.config.optimizer_extensions.push_back(LanceOptimizerExtension());
}

} // namespace duckdb
