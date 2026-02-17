// ANN index scan optimizer for LANCE indexes.
// Rewrites: ORDER BY array_distance(col, query) LIMIT k → Lance index scan.

#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/scan_state.hpp"
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
};

struct LanceIndexScanGlobalState : public GlobalTableFunctionState {
	vector<pair<row_t, float>> results;
	idx_t offset = 0;
	vector<StorageIndex> storage_ids;

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

	// Compute storage IDs from the planner's column_ids
	for (auto &col_id : input.column_ids) {
		state->storage_ids.push_back(StorageIndex(col_id));
	}

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
	storage.Fetch(transaction, output, state.storage_ids, row_ids_vec, batch_size, fetch_state);

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

		// Must be a constant limit (not percentage or expression)
		if (limit_op.limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
			return;
		}
		auto limit_val = limit_op.limit_val.GetConstantValue();

		// Bail if OFFSET is present — index scan returns top-k, can't skip rows
		if (limit_op.offset_val.Type() == LimitNodeType::CONSTANT_VALUE && limit_op.offset_val.GetConstantValue() > 0) {
			return;
		}

		if (op->children.empty() || op->children[0]->type != LogicalOperatorType::LOGICAL_ORDER_BY) {
			return;
		}
		auto &order_op = op->children[0]->Cast<LogicalOrder>();

		if (order_op.orders.size() != 1) {
			return;
		}

		// Only rewrite ASC ordering — DESC wants farthest vectors, not nearest
		if (order_op.orders[0].type == OrderType::DESCENDING) {
			return;
		}

		auto &order_expr = order_op.orders[0].expression;

		if (!IsArrayDistanceFunction(*order_expr)) {
			return;
		}

		auto &func_expr = order_expr->Cast<BoundFunctionExpression>();
		if (func_expr.children.size() != 2) {
			return;
		}

		// Second argument must be a constant (the query vector)
		if (func_expr.children[1]->type != ExpressionType::VALUE_CONSTANT) {
			return;
		}
		auto &query_const = func_expr.children[1]->Cast<BoundConstantExpression>();
		auto &qval = query_const.value;

		// Extract query vector floats from ARRAY or LIST constant
		vector<float> query_vector;
		if (qval.type().id() == LogicalTypeId::ARRAY) {
			auto &children = ArrayValue::GetChildren(qval);
			for (auto &c : children) {
				query_vector.push_back(c.GetValue<float>());
			}
		} else if (qval.type().id() == LogicalTypeId::LIST) {
			auto &children = ListValue::GetChildren(qval);
			for (auto &c : children) {
				query_vector.push_back(c.GetValue<float>());
			}
		} else {
			return;
		}
		if (query_vector.empty()) {
			return;
		}

		// Walk from ORDER's child to find the GET: order -> [projection ->] GET
		auto *child_of_order = order_op.children[0].get();
		LogicalGet *get_ptr = nullptr;
		bool has_projection = false;

		if (child_of_order->type == LogicalOperatorType::LOGICAL_GET) {
			get_ptr = &child_of_order->Cast<LogicalGet>();
		} else if (child_of_order->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			has_projection = true;
			if (!child_of_order->children.empty() &&
			    child_of_order->children[0]->type == LogicalOperatorType::LOGICAL_GET) {
				get_ptr = &child_of_order->children[0]->Cast<LogicalGet>();
			}
		}
		if (!get_ptr) {
			return;
		}

		auto &get = *get_ptr;

		// Verify this is a DuckDB table with LANCE indexes
		auto table_entry = get.GetTable();
		if (!table_entry || !table_entry->IsDuckTable()) {
			return;
		}
		auto &duck_table = table_entry->Cast<DuckTableEntry>();
		auto &storage = duck_table.GetStorage();
		auto &table_info = *storage.GetDataTableInfo();
		auto &indexes = table_info.GetIndexes();

		indexes.Bind(input.context, table_info, LanceIndex::TYPE_NAME);

		// Resolve which physical column the distance function references
		column_t target_column = DConstants::INVALID_INDEX;
		auto &first_child = func_expr.children[0];
		if (first_child->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &col_ref = first_child->Cast<BoundColumnRefExpression>();
			auto col_idx = col_ref.binding.column_index;
			auto &col_ids = get.GetColumnIds();
			if (col_idx < col_ids.size()) {
				target_column = col_ids[col_idx].GetPrimaryIndex();
			}
		}

		// Find LANCE index whose column_ids match the target column
		string found_index;
		indexes.Scan([&](Index &idx) {
			if (idx.GetIndexType() != LanceIndex::TYPE_NAME) {
				return false;
			}
			auto &idx_cols = idx.GetColumnIds();
			for (auto &col : idx_cols) {
				if (col == target_column) {
					found_index = idx.GetIndexName();
					return true;
				}
			}
			return false;
		});
		if (found_index.empty()) {
			return;
		}

		// Build bind data for the replacement scan
		auto bind_data = make_uniq<LanceIndexScanBindData>();
		bind_data->table_entry = &duck_table;
		bind_data->index_name = found_index;
		bind_data->limit = limit_val;
		bind_data->vector_size = query_vector.size();
		bind_data->query_vector = make_unsafe_uniq_array<float>(query_vector.size());
		memcpy(bind_data->query_vector.get(), query_vector.data(), query_vector.size() * sizeof(float));

		// Create the replacement table function
		TableFunction scan_func("lance_index_scan", {}, LanceIndexScanScan, LanceIndexScanBind, LanceIndexScanInit);

		// Reuse the original GET's table_index so column references from above still resolve
		auto new_get = make_uniq<LogicalGet>(get.table_index, scan_func, std::move(bind_data),
		                                     get.returned_types, get.names);
		new_get->GetMutableColumnIds() = get.GetColumnIds();
		new_get->projection_ids = get.projection_ids;

		// Replace the LIMIT → ORDER → ... subtree
		if (has_projection) {
			// LIMIT → ORDER → PROJ → GET  →  PROJ → new GET
			auto proj = std::move(order_op.children[0]);
			proj->children[0] = std::move(new_get);
			op = std::move(proj);
		} else {
			// LIMIT → ORDER → GET  →  new GET
			op = std::move(new_get);
		}
	}
};

void RegisterLanceOptimizer(DatabaseInstance &db) {
	db.config.optimizer_extensions.push_back(LanceOptimizerExtension());
}

} // namespace duckdb
