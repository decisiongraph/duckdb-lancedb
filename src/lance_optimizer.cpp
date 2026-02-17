// ANN index scan optimizer for LANCE indexes.
// Rewrites: ORDER BY array_distance(col, query) LIMIT k → Lance index scan.
// Supports filter pushdown: WHERE predicates are converted to Lance SQL and pushed to LanceDB.

#include "lancedb_extension.hpp"
#include "lance_index.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
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
	string predicate;
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

static unique_ptr<GlobalTableFunctionState> LanceIndexScanInit(ClientContext &context, TableFunctionInitInput &input) {
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
		state->results = lance_idx.Search(bind_data.query_vector.get(), static_cast<int32_t>(bind_data.vector_size),
		                                  static_cast<int32_t>(bind_data.limit), bind_data.predicate);
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
	// Note: storage.Fetch() already sets the correct cardinality (skipping MVCC-invisible rows).
}

// ========================================
// Expression-to-Lance predicate converter
// ========================================

// Resolve a BoundColumnRefExpression to its table column name via the GET's column_ids.
static bool ResolveColumnName(const BoundColumnRefExpression &col_ref, const LogicalGet &get, string &out_name) {
	auto col_idx = col_ref.binding.column_index;
	auto &col_ids = get.GetColumnIds();
	if (col_idx >= col_ids.size()) {
		return false;
	}
	auto physical_col = col_ids[col_idx].GetPrimaryIndex();

	// Map physical column id to column name
	auto table_entry = get.GetTable();
	if (!table_entry) {
		return false;
	}
	auto &columns = table_entry->GetColumns();
	if (physical_col >= columns.PhysicalColumnCount()) {
		return false;
	}
	out_name = columns.GetColumn(PhysicalIndex(physical_col)).GetName();
	return true;
}

// Format a constant value as a Lance SQL literal.
static bool FormatConstant(const Value &val, string &out) {
	if (val.IsNull()) {
		out = "NULL";
		return true;
	}
	switch (val.type().id()) {
	case LogicalTypeId::VARCHAR:
		// Escape single quotes
		{
			auto str = val.GetValue<string>();
			string escaped;
			escaped.reserve(str.size() + 2);
			escaped += '\'';
			for (char c : str) {
				if (c == '\'') {
					escaped += "''";
				} else {
					escaped += c;
				}
			}
			escaped += '\'';
			out = escaped;
		}
		return true;
	case LogicalTypeId::INTEGER:
		out = std::to_string(val.GetValue<int32_t>());
		return true;
	case LogicalTypeId::BIGINT:
		out = std::to_string(val.GetValue<int64_t>());
		return true;
	case LogicalTypeId::FLOAT:
		out = val.ToString();
		return true;
	case LogicalTypeId::DOUBLE:
		out = val.ToString();
		return true;
	case LogicalTypeId::BOOLEAN:
		out = val.GetValue<bool>() ? "true" : "false";
		return true;
	default:
		return false;
	}
}

// Convert a comparison ExpressionType to SQL operator string.
static const char *ComparisonOperator(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
		return "=";
	case ExpressionType::COMPARE_NOTEQUAL:
		return "!=";
	case ExpressionType::COMPARE_LESSTHAN:
		return "<";
	case ExpressionType::COMPARE_GREATERTHAN:
		return ">";
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return "<=";
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ">=";
	default:
		return nullptr;
	}
}

// Recursively convert a DuckDB expression to a Lance predicate string.
// Returns true if conversion succeeded (expression is pushable).
static bool ExpressionToLancePredicate(const Expression &expr, const LogicalGet &get, string &out) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COMPARISON: {
		auto &cmp = expr.Cast<BoundComparisonExpression>();
		auto *op_str = ComparisonOperator(expr.type);
		if (!op_str) {
			return false;
		}

		// column op constant  OR  constant op column
		string left_sql, right_sql;
		bool left_ok = false, right_ok = false;

		if (cmp.left->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			left_ok = ResolveColumnName(cmp.left->Cast<BoundColumnRefExpression>(), get, left_sql);
		} else if (cmp.left->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			left_ok = FormatConstant(cmp.left->Cast<BoundConstantExpression>().value, left_sql);
		}

		if (cmp.right->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			right_ok = ResolveColumnName(cmp.right->Cast<BoundColumnRefExpression>(), get, right_sql);
		} else if (cmp.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			right_ok = FormatConstant(cmp.right->Cast<BoundConstantExpression>().value, right_sql);
		}

		if (!left_ok || !right_ok) {
			return false;
		}

		out = left_sql + " " + op_str + " " + right_sql;
		return true;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		const char *join_op = (expr.type == ExpressionType::CONJUNCTION_AND) ? " AND " : " OR ";

		string result;
		for (idx_t i = 0; i < conj.children.size(); i++) {
			string child_sql;
			if (!ExpressionToLancePredicate(*conj.children[i], get, child_sql)) {
				return false;
			}
			if (i > 0) {
				result += join_op;
			}
			// Wrap in parens for clarity with OR
			if (expr.type == ExpressionType::CONJUNCTION_OR) {
				result += "(" + child_sql + ")";
			} else {
				result += child_sql;
			}
		}
		out = result;
		return true;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = expr.Cast<BoundOperatorExpression>();
		switch (expr.type) {
		case ExpressionType::OPERATOR_IS_NULL:
		case ExpressionType::OPERATOR_IS_NOT_NULL: {
			if (op_expr.children.size() != 1 ||
			    op_expr.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			string col_name;
			if (!ResolveColumnName(op_expr.children[0]->Cast<BoundColumnRefExpression>(), get, col_name)) {
				return false;
			}
			out = col_name + (expr.type == ExpressionType::OPERATOR_IS_NULL ? " IS NULL" : " IS NOT NULL");
			return true;
		}
		case ExpressionType::OPERATOR_NOT: {
			if (op_expr.children.size() != 1) {
				return false;
			}
			string child_sql;
			if (!ExpressionToLancePredicate(*op_expr.children[0], get, child_sql)) {
				return false;
			}
			out = "NOT (" + child_sql + ")";
			return true;
		}
		case ExpressionType::COMPARE_IN:
		case ExpressionType::COMPARE_NOT_IN: {
			// children[0] = column, children[1..] = constants
			if (op_expr.children.size() < 2 ||
			    op_expr.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			string col_name;
			if (!ResolveColumnName(op_expr.children[0]->Cast<BoundColumnRefExpression>(), get, col_name)) {
				return false;
			}
			string values;
			for (idx_t i = 1; i < op_expr.children.size(); i++) {
				if (op_expr.children[i]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
					return false;
				}
				string val_str;
				if (!FormatConstant(op_expr.children[i]->Cast<BoundConstantExpression>().value, val_str)) {
					return false;
				}
				if (i > 1) {
					values += ", ";
				}
				values += val_str;
			}
			out = col_name + (expr.type == ExpressionType::COMPARE_IN ? " IN (" : " NOT IN (") + values + ")";
			return true;
		}
		default:
			return false;
		}
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between = expr.Cast<BoundBetweenExpression>();
		if (between.input->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		string col_name;
		if (!ResolveColumnName(between.input->Cast<BoundColumnRefExpression>(), get, col_name)) {
			return false;
		}
		if (between.lower->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT ||
		    between.upper->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
			return false;
		}
		string lower_str, upper_str;
		if (!FormatConstant(between.lower->Cast<BoundConstantExpression>().value, lower_str) ||
		    !FormatConstant(between.upper->Cast<BoundConstantExpression>().value, upper_str)) {
			return false;
		}
		auto lower_op = between.lower_inclusive ? ">=" : ">";
		auto upper_op = between.upper_inclusive ? "<=" : "<";
		out = col_name + " " + lower_op + " " + lower_str + " AND " + col_name + " " + upper_op + " " + upper_str;
		return true;
	}
	default:
		return false;
	}
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

// Map DuckDB distance function name to Lance metric string.
static string DistanceFunctionToMetric(const string &func_name) {
	if (func_name == "array_distance") {
		return "l2";
	}
	if (func_name == "array_cosine_distance") {
		return "cosine";
	}
	if (func_name == "array_inner_product") {
		return "dot";
	}
	return "";
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

		// Walk from ORDER's child to find the GET, optionally through FILTER and/or PROJECTION.
		// Possible trees after DuckDB's own optimizers:
		//   ORDER → GET
		//   ORDER → PROJ → GET
		//   ORDER → FILTER → GET
		//   ORDER → FILTER → PROJ → GET
		auto *walk = order_op.children[0].get();
		LogicalGet *get_ptr = nullptr;
		bool has_projection = false;
		LogicalFilter *filter_ptr = nullptr;

		if (walk->type == LogicalOperatorType::LOGICAL_FILTER) {
			filter_ptr = &walk->Cast<LogicalFilter>();
			walk = walk->children.empty() ? nullptr : walk->children[0].get();
		}

		if (walk && walk->type == LogicalOperatorType::LOGICAL_GET) {
			get_ptr = &walk->Cast<LogicalGet>();
		} else if (walk && walk->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			has_projection = true;
			if (!walk->children.empty() && walk->children[0]->type == LogicalOperatorType::LOGICAL_GET) {
				get_ptr = &walk->children[0]->Cast<LogicalGet>();
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

		// Find LANCE index whose column_ids and metric match the query
		auto expected_metric = DistanceFunctionToMetric(func_expr.function.name);
		string found_index;
		indexes.Scan([&](Index &idx) {
			if (idx.GetIndexType() != LanceIndex::TYPE_NAME) {
				return false;
			}
			auto &idx_cols = idx.GetColumnIds();
			bool col_match = false;
			for (auto &col : idx_cols) {
				if (col == target_column) {
					col_match = true;
					break;
				}
			}
			if (!col_match) {
				return false;
			}
			// Verify metric compatibility
			auto &lance_idx = idx.Cast<LanceIndex>();
			auto &idx_metric = lance_idx.GetMetric();
			// "dot" and "ip" are equivalent names for inner product
			bool metric_match = (idx_metric == expected_metric) || (idx_metric == "ip" && expected_metric == "dot") ||
			                    (idx_metric == "dot" && expected_metric == "ip");
			if (!metric_match) {
				return false;
			}
			found_index = idx.GetIndexName();
			return true;
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

		// Try to push filter predicates to Lance
		bool filter_fully_pushed = false;
		if (filter_ptr && !filter_ptr->expressions.empty()) {
			// Split conjunctions into individual predicates
			LogicalFilter::SplitPredicates(filter_ptr->expressions);

			vector<unique_ptr<Expression>> pushable;
			vector<unique_ptr<Expression>> remaining;

			for (auto &expr : filter_ptr->expressions) {
				string pred_sql;
				if (ExpressionToLancePredicate(*expr, get, pred_sql)) {
					// Accumulate into Lance predicate
					if (!bind_data->predicate.empty()) {
						bind_data->predicate += " AND ";
					}
					bind_data->predicate += pred_sql;
					pushable.push_back(std::move(expr));
				} else {
					remaining.push_back(std::move(expr));
				}
			}

			filter_fully_pushed = remaining.empty();

			if (!filter_fully_pushed) {
				// Keep remaining predicates in the FILTER node
				filter_ptr->expressions = std::move(remaining);
			}
		}

		// Create the replacement table function
		TableFunction scan_func("lance_index_scan", {}, LanceIndexScanScan, LanceIndexScanBind, LanceIndexScanInit);

		// Reuse the original GET's table_index so column references from above still resolve
		auto new_get =
		    make_uniq<LogicalGet>(get.table_index, scan_func, std::move(bind_data), get.returned_types, get.names);
		new_get->GetMutableColumnIds() = get.GetColumnIds();
		new_get->projection_ids = get.projection_ids;

		// Replace the LIMIT → ORDER → [FILTER →] [PROJ →] GET subtree
		if (filter_ptr && !filter_fully_pushed) {
			// Partial pushdown: keep FILTER, replace its child subtree with new GET
			if (has_projection) {
				// FILTER → PROJ → GET  →  FILTER → PROJ → new GET
				auto *proj_node = filter_ptr->children[0].get();
				proj_node->children[0] = std::move(new_get);
				// Move FILTER (with PROJ child) out to replace LIMIT
				op = std::move(order_op.children[0]);
			} else {
				// FILTER → GET  →  FILTER → new GET
				filter_ptr->children[0] = std::move(new_get);
				op = std::move(order_op.children[0]);
			}
		} else if (has_projection) {
			// LIMIT → ORDER → [FILTER →] PROJ → GET  →  PROJ → new GET
			unique_ptr<LogicalOperator> proj;
			if (filter_ptr) {
				// filter fully pushed: FILTER → PROJ → GET
				proj = std::move(filter_ptr->children[0]);
			} else {
				proj = std::move(order_op.children[0]);
			}
			proj->children[0] = std::move(new_get);
			op = std::move(proj);
		} else {
			// LIMIT → ORDER → [FILTER →] GET  →  new GET
			op = std::move(new_get);
		}
	}
};

void RegisterLanceOptimizer(DatabaseInstance &db) {
	db.config.optimizer_extensions.push_back(LanceOptimizerExtension());
}

} // namespace duckdb
