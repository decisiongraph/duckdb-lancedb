#include "lance_index.hpp"
#include "lancedb_extension.hpp"
#include "rust_ffi.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <cstring>
#include <cstdlib>
#include <unistd.h>

namespace duckdb {

// Sanitize index name for safe use in filesystem paths
static string SanitizeIndexName(const string &name) {
	string result;
	result.reserve(name.size());
	for (char c : name) {
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') {
			result += c;
		} else {
			result += '_';
		}
	}
	if (result.empty()) {
		result = "lance_idx";
	}
	return result;
}

// Generate a unique temp directory path for in-memory DBs
static string MakeUniqueTempPath(const string &sanitized_name) {
	const char *tmp_dir = std::getenv("TMPDIR");
	if (!tmp_dir) {
		tmp_dir = "/tmp";
	}
	auto pid = std::to_string(getpid());
	// Use address of a stack variable as a cheap unique suffix
	int stack_var;
	auto addr = reinterpret_cast<uintptr_t>(&stack_var);
	return string(tmp_dir) + "/duckdb_lance_" + pid + "_" + sanitized_name + "_" + std::to_string(addr);
}

// ========================================
// LinkedBlock storage (metadata only for Lance)
// ========================================

struct LinkedBlock {
	static constexpr idx_t BLOCK_SIZE = Storage::DEFAULT_BLOCK_SIZE - sizeof(validity_t);
	static constexpr idx_t BLOCK_DATA_SIZE = BLOCK_SIZE - sizeof(IndexPointer);

	IndexPointer next_block;
	char data[BLOCK_DATA_SIZE] = {0};
};

class LinkedBlockWriter {
public:
	LinkedBlockWriter(FixedSizeAllocator &allocator, IndexPointer root)
	    : allocator_(allocator), root_(root), current_(root), pos_(0) {
	}

	void Reset() {
		current_ = root_;
		pos_ = 0;
	}

	idx_t Write(const uint8_t *buffer, idx_t length) {
		idx_t written = 0;
		while (written < length) {
			auto block = allocator_.Get<LinkedBlock>(current_, true);
			auto to_write = MinValue<idx_t>(length - written, LinkedBlock::BLOCK_DATA_SIZE - pos_);
			memcpy(block->data + pos_, buffer + written, to_write);
			written += to_write;
			pos_ += to_write;
			if (pos_ == LinkedBlock::BLOCK_DATA_SIZE) {
				pos_ = 0;
				if (block->next_block.Get() == 0) {
					block->next_block = allocator_.New();
				}
				current_ = block->next_block;
			}
		}
		return written;
	}

private:
	FixedSizeAllocator &allocator_;
	IndexPointer root_;
	IndexPointer current_;
	idx_t pos_;
};

class LinkedBlockReader {
public:
	LinkedBlockReader(FixedSizeAllocator &allocator, IndexPointer root)
	    : allocator_(allocator), current_(root), pos_(0), exhausted_(false) {
	}

	idx_t Read(uint8_t *buffer, idx_t length) {
		idx_t total_read = 0;
		while (total_read < length && !exhausted_) {
			auto block = allocator_.Get<LinkedBlock>(current_, false);
			auto to_read = MinValue<idx_t>(length - total_read, LinkedBlock::BLOCK_DATA_SIZE - pos_);
			memcpy(buffer + total_read, block->data + pos_, to_read);
			total_read += to_read;
			pos_ += to_read;
			if (pos_ == LinkedBlock::BLOCK_DATA_SIZE) {
				pos_ = 0;
				if (block->next_block.Get() == 0) {
					exhausted_ = true;
				} else {
					current_ = block->next_block;
				}
			}
		}
		return total_read;
	}

private:
	FixedSizeAllocator &allocator_;
	IndexPointer current_;
	idx_t pos_;
	bool exhausted_;
};

// ========================================
// LanceIndex implementation
// ========================================

LanceIndex::LanceIndex(const string &name, IndexConstraintType constraint_type, const vector<column_t> &column_ids,
                       TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
                       AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
                       const IndexStorageInfo &info)
    : BoundIndex(name, TYPE_NAME, constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("LANCE indexes do not support unique/primary key constraints");
	}

	// Parse options
	for (auto &kv : options) {
		if (kv.first == "metric") {
			metric_ = kv.second.ToString();
		} else if (kv.first == "nprobes") {
			nprobes_ = kv.second.GetValue<int32_t>();
		} else if (kv.first == "refine_factor") {
			refine_factor_ = kv.second.GetValue<int32_t>();
		}
	}

	// Detect dimension from expression type
	if (!unbound_expressions.empty()) {
		auto &type = unbound_expressions[0]->return_type;
		if (type.id() == LogicalTypeId::ARRAY) {
			dimension_ = static_cast<int32_t>(ArrayType::GetSize(type));
		}
	}

	// Initialize block allocator
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	block_allocator_ = make_uniq<FixedSizeAllocator>(LinkedBlock::BLOCK_SIZE, block_manager);

	if (info.IsValid()) {
		LoadFromStorage(info);
	}
}

LanceIndex::~LanceIndex() {
	if (rust_handle_) {
		LanceFreeDetached(rust_handle_);
		rust_handle_ = nullptr;
	}
}

string LanceIndex::GetLancePath() {
	if (!lance_path_.empty()) {
		return lance_path_;
	}
	auto sanitized = SanitizeIndexName(name);
	auto &storage_manager = db.GetStorageManager();
	auto db_path = storage_manager.GetDBPath();
	if (db_path.empty()) {
		// In-memory DB: use unique temp directory to avoid collisions
		lance_path_ = MakeUniqueTempPath(sanitized);
	} else {
		lance_path_ = db_path + ".lance/" + sanitized;
	}
	return lance_path_;
}

unique_ptr<BoundIndex> LanceIndex::Create(CreateIndexInput &input) {
	return make_uniq<LanceIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
	                             input.unbound_expressions, input.db, input.options, input.storage_info);
}

PhysicalOperator &LanceIndex::CreatePlan(PlanIndexInput &input) {
	auto &op = input.op;
	auto &planner = input.planner;

	// Validate: single FLOAT[N] column
	if (op.unbound_expressions.size() != 1) {
		throw InvalidInputException("LANCE index requires exactly one column");
	}
	auto &type = op.unbound_expressions[0]->return_type;
	if (type.id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
		throw InvalidInputException("LANCE index column must be FLOAT[N]");
	}

	// PROJECTION on indexed column + row_id
	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (idx_t i = 0; i < op.expressions.size(); i++) {
		new_column_types.push_back(op.expressions[i]->return_type);
		select_list.push_back(std::move(op.expressions[i]));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(
	    make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, op.info->scan_types.size() - 1));

	auto &proj =
	    planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), op.estimated_cardinality);
	proj.children.push_back(input.table_scan);

	auto &create_idx = planner.Make<PhysicalCreateLanceIndex>(op, op.table, op.info->column_ids,
	                                                          std::move(op.info),
	                                                          std::move(op.unbound_expressions),
	                                                          op.estimated_cardinality,
	                                                          std::move(op.alter_table_info));
	create_idx.children.push_back(proj);
	return create_idx;
}

// ========================================
// Append / Insert / Delete
// ========================================

ErrorData LanceIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	auto count = entries.size();
	if (count == 0) {
		return ErrorData {};
	}

	DataChunk expr_chunk;
	expr_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(entries, expr_chunk);

	if (!rust_handle_) {
		auto lance_path = GetLancePath();
		auto table_name = SanitizeIndexName(name);
		rust_handle_ = LanceCreateDetached(lance_path, dimension_, metric_, table_name);
	}

	auto &vec_col = expr_chunk.data[0];
	auto &array_child = ArrayVector::GetEntry(vec_col);
	auto array_size = ArrayType::GetSize(vec_col.GetType());
	auto child_data = FlatVector::GetData<float>(array_child);

	UnifiedVectorFormat rowid_format;
	row_identifiers.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	vector<int64_t> labels(count);
	auto n = LanceDetachedAddBatch(rust_handle_, child_data,
	                               static_cast<int32_t>(count),
	                               dimension_, labels.data());

	for (idx_t i = 0; i < static_cast<idx_t>(n); i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];
		auto label = labels[i];

		if (static_cast<idx_t>(label) >= label_to_rowid_.size()) {
			label_to_rowid_.resize(label + 1, -1);
		}
		label_to_rowid_[label] = row_id;
		rowid_to_label_[row_id] = label;
	}

	is_dirty_ = true;
	return ErrorData {};
}

ErrorData LanceIndex::Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) {
	return Append(lock, data, row_ids);
}

void LanceIndex::Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	auto count = entries.size();
	if (count == 0) {
		return;
	}

	UnifiedVectorFormat rowid_format;
	row_identifiers.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	// Collect labels to delete in a single batch
	vector<int64_t> labels_to_delete;
	labels_to_delete.reserve(count);

	for (idx_t i = 0; i < count; i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];

		auto it = rowid_to_label_.find(row_id);
		if (it != rowid_to_label_.end()) {
			deleted_labels_.insert(it->second);
			labels_to_delete.push_back(it->second);
			rowid_to_label_.erase(it);
		}
	}

	if (rust_handle_ && !labels_to_delete.empty()) {
		LanceDetachedDeleteBatch(rust_handle_, labels_to_delete.data(),
		                         static_cast<int32_t>(labels_to_delete.size()));
	}

	is_dirty_ = true;
}

void LanceIndex::CommitDrop(IndexLock &lock) {
	if (rust_handle_) {
		LanceFreeDetached(rust_handle_);
		rust_handle_ = nullptr;
	}
	// Clean up Lance directory
	auto lance_path = GetLancePath();
	auto &fs = FileSystem::GetFileSystem(db.GetDatabase());
	fs.RemoveDirectory(lance_path);
}

// ========================================
// Search
// ========================================

vector<pair<row_t, float>> LanceIndex::Search(const float *query, int32_t dimension, int32_t k) {
	if (!rust_handle_ || dimension != dimension_) {
		return {};
	}

	// Lance already excludes deleted vectors (Delete() calls LanceDetachedDeleteBatch),
	// so no C++ tombstone filtering or retry loop needed.
	vector<int64_t> labels(k);
	vector<float> distances(k);
	auto n = LanceDetachedSearch(rust_handle_, query, dimension, k, nprobes_, refine_factor_,
	                             labels.data(), distances.data());

	vector<pair<row_t, float>> results;
	results.reserve(n);
	for (int32_t i = 0; i < n; i++) {
		auto label = labels[i];
		if (label >= 0 && label < static_cast<int64_t>(label_to_rowid_.size())) {
			results.emplace_back(label_to_rowid_[label], distances[i]);
		}
	}

	return results;
}

void LanceIndex::CreateAnnIndex(int32_t num_partitions, int32_t num_sub_vectors) {
	if (!rust_handle_) {
		throw IOException("Lance index not initialized");
	}
	LanceDetachedCreateIndex(rust_handle_, num_partitions, num_sub_vectors);
}

// ========================================
// Persistence (metadata only — Lance handles vector data)
// ========================================

void LanceIndex::PersistToDisk() {
	if (!is_dirty_ || !rust_handle_) {
		return;
	}

	if (root_block_ptr_.Get() == 0) {
		root_block_ptr_ = block_allocator_->New();
	}

	// Serialize Lance metadata
	auto meta = LanceDetachedSerializeMeta(rust_handle_);

	LinkedBlockWriter writer(*block_allocator_, root_block_ptr_);
	writer.Reset();

	// Write Lance metadata
	uint64_t meta_len = meta.len;
	writer.Write(reinterpret_cast<const uint8_t *>(&meta_len), sizeof(uint64_t));
	writer.Write(meta.data, meta.len);

	// Write label → row_id mappings
	uint64_t num_mappings = label_to_rowid_.size();
	writer.Write(reinterpret_cast<const uint8_t *>(&num_mappings), sizeof(uint64_t));
	if (num_mappings > 0) {
		writer.Write(reinterpret_cast<const uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	// Write deleted labels
	uint64_t num_tombstones = deleted_labels_.size();
	writer.Write(reinterpret_cast<const uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<int64_t> tombstone_vec(deleted_labels_.begin(), deleted_labels_.end());
		writer.Write(reinterpret_cast<const uint8_t *>(tombstone_vec.data()), num_tombstones * sizeof(int64_t));
	}

	// Write parameters
	writer.Write(reinterpret_cast<const uint8_t *>(&dimension_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&nprobes_), sizeof(int32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(&refine_factor_), sizeof(int32_t));
	uint32_t metric_len_val = static_cast<uint32_t>(metric_.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&metric_len_val), sizeof(uint32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(metric_.data()), metric_len_val);

	// Write Lance path
	auto lance_path = GetLancePath();
	uint32_t path_len = static_cast<uint32_t>(lance_path.size());
	writer.Write(reinterpret_cast<const uint8_t *>(&path_len), sizeof(uint32_t));
	writer.Write(reinterpret_cast<const uint8_t *>(lance_path.data()), path_len);

	LanceFreeBytes(meta);
	is_dirty_ = false;
}

void LanceIndex::LoadFromStorage(const IndexStorageInfo &info) {
	if (!info.IsValid() || info.allocator_infos.empty()) {
		return;
	}

	root_block_ptr_.Set(info.root);
	block_allocator_->Init(info.allocator_infos[0]);

	LinkedBlockReader reader(*block_allocator_, root_block_ptr_);

	// Read Lance metadata
	uint64_t meta_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&meta_len), sizeof(uint64_t));

	vector<uint8_t> meta_data(meta_len);
	reader.Read(meta_data.data(), meta_len);

	// Read label → row_id mappings
	uint64_t num_mappings = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_mappings), sizeof(uint64_t));
	label_to_rowid_.resize(num_mappings);
	if (num_mappings > 0) {
		reader.Read(reinterpret_cast<uint8_t *>(label_to_rowid_.data()), num_mappings * sizeof(row_t));
	}

	// Read deleted labels
	uint64_t num_tombstones = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&num_tombstones), sizeof(uint64_t));
	if (num_tombstones > 0) {
		vector<int64_t> tombstones(num_tombstones);
		reader.Read(reinterpret_cast<uint8_t *>(tombstones.data()), num_tombstones * sizeof(int64_t));
		deleted_labels_.insert(tombstones.begin(), tombstones.end());
	}

	// Read parameters
	reader.Read(reinterpret_cast<uint8_t *>(&dimension_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&nprobes_), sizeof(int32_t));
	reader.Read(reinterpret_cast<uint8_t *>(&refine_factor_), sizeof(int32_t));
	uint32_t metric_len_val = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&metric_len_val), sizeof(uint32_t));
	vector<char> metric_buf(metric_len_val);
	reader.Read(reinterpret_cast<uint8_t *>(metric_buf.data()), metric_len_val);
	metric_.assign(metric_buf.data(), metric_len_val);

	// Read Lance path
	uint32_t path_len = 0;
	reader.Read(reinterpret_cast<uint8_t *>(&path_len), sizeof(uint32_t));
	vector<char> path_buf(path_len);
	reader.Read(reinterpret_cast<uint8_t *>(path_buf.data()), path_len);
	string lance_path(path_buf.data(), path_len);

	// Rebuild reverse mappings
	for (size_t i = 0; i < label_to_rowid_.size(); i++) {
		if (deleted_labels_.count(static_cast<int64_t>(i)) == 0) {
			rowid_to_label_[label_to_rowid_[i]] = static_cast<int64_t>(i);
		}
	}

	// Reopen Lance dataset
	rust_handle_ = LanceDetachedDeserializeMeta(lance_path, meta_data.data(), meta_data.size());
	is_dirty_ = false;
}

IndexStorageInfo LanceIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr_.Get();

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	block_allocator_->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();
	info.allocator_infos.push_back(block_allocator_->GetInfo());

	return info;
}

IndexStorageInfo LanceIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	IndexStorageInfo info;
	info.name = name;
	info.root = root_block_ptr_.Get();
	info.buffers.push_back(block_allocator_->InitSerializationToWAL());
	info.allocator_infos.push_back(block_allocator_->GetInfo());

	return info;
}

idx_t LanceIndex::GetInMemorySize(IndexLock &state) {
	idx_t size = sizeof(LanceIndex);
	size += label_to_rowid_.size() * sizeof(row_t);
	size += rowid_to_label_.size() * (sizeof(row_t) + sizeof(int64_t));
	return size;
}

bool LanceIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	auto &other = other_index.Cast<LanceIndex>();
	if (!other.rust_handle_ || !rust_handle_) {
		is_dirty_ = true;
		return true;
	}

	// Bulk export all vectors from other index in a single FFI call
	int64_t other_vec_count = 0;
	LanceDetachedGetAllVectors(other.rust_handle_, nullptr, nullptr, &other_vec_count);

	if (other_vec_count <= 0) {
		is_dirty_ = true;
		return true;
	}

	vector<int64_t> other_labels(other_vec_count);
	vector<float> other_vectors(other_vec_count * dimension_);
	LanceDetachedGetAllVectors(other.rust_handle_, other_labels.data(), other_vectors.data(), &other_vec_count);

	// Filter out tombstoned vectors and collect live ones
	vector<float> live_vectors;
	vector<row_t> live_rowids;

	for (int64_t i = 0; i < other_vec_count; i++) {
		auto label = other_labels[i];
		if (other.deleted_labels_.count(label) > 0) {
			continue;
		}
		if (label >= static_cast<int64_t>(other.label_to_rowid_.size())) {
			continue;
		}

		auto vec_start = i * dimension_;
		live_vectors.insert(live_vectors.end(), other_vectors.begin() + vec_start,
		                    other_vectors.begin() + vec_start + dimension_);
		live_rowids.push_back(other.label_to_rowid_[label]);
	}

	// Single batch insert
	if (!live_rowids.empty()) {
		auto num = static_cast<int32_t>(live_rowids.size());
		vector<int64_t> new_labels(num);
		LanceDetachedAddBatch(rust_handle_, live_vectors.data(), num, dimension_, new_labels.data());

		for (idx_t i = 0; i < static_cast<idx_t>(num); i++) {
			auto row_id = live_rowids[i];
			auto label = new_labels[i];

			if (static_cast<idx_t>(label) >= label_to_rowid_.size()) {
				label_to_rowid_.resize(label + 1, -1);
			}
			label_to_rowid_[label] = row_id;
			rowid_to_label_[row_id] = label;
		}
	}

	is_dirty_ = true;
	return true;
}

void LanceIndex::Vacuum(IndexLock &state) {
	if (deleted_labels_.empty() || !rust_handle_) {
		return;
	}
	LanceDetachedCompact(rust_handle_);
	deleted_labels_.clear();
	is_dirty_ = true;
}

string LanceIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	if (only_verify) {
		return "ok";
	}
	return "LanceIndex(dim=" + std::to_string(dimension_) + ", metric=" + metric_ +
	       ", vectors=" + std::to_string(GetVectorCount()) + ")";
}

void LanceIndex::VerifyAllocations(IndexLock &state) {
}
void LanceIndex::VerifyBuffers(IndexLock &l) {
}

string LanceIndex::GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
                                                 DataChunk &input) {
	return "LANCE indexes do not support constraints";
}

// ========================================
// PhysicalCreateLanceIndex
// ========================================

struct CreateLanceGlobalSinkState : public GlobalSinkState {
	LanceHandle rust_handle = nullptr;
	vector<row_t> label_to_rowid;
	unordered_map<row_t, int64_t> rowid_to_label;
	int32_t dimension = 0;
	string metric = "l2";
	int32_t nprobes = 20;
	int32_t refine_factor = 1;
	string lance_path;

	~CreateLanceGlobalSinkState() override {
		if (rust_handle) {
			LanceFreeDetached(rust_handle);
		}
	}
};

PhysicalCreateLanceIndex::PhysicalCreateLanceIndex(PhysicalPlan &physical_plan, LogicalOperator &op,
                                                   TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                                   unique_ptr<CreateIndexInfo> info_p,
                                                   vector<unique_ptr<Expression>> unbound_expressions_p,
                                                   idx_t estimated_cardinality,
                                                   unique_ptr<AlterTableInfo> alter_table_info_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::CREATE_INDEX, op.types, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info_p)),
      unbound_expressions(std::move(unbound_expressions_p)),
      alter_table_info(std::move(alter_table_info_p)) {
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

unique_ptr<GlobalSinkState> PhysicalCreateLanceIndex::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_uniq<CreateLanceGlobalSinkState>();

	auto &type = unbound_expressions[0]->return_type;
	state->dimension = static_cast<int32_t>(ArrayType::GetSize(type));

	for (auto &kv : info->options) {
		if (kv.first == "metric") {
			state->metric = kv.second.ToString();
		} else if (kv.first == "nprobes") {
			state->nprobes = kv.second.GetValue<int32_t>();
		} else if (kv.first == "refine_factor") {
			state->refine_factor = kv.second.GetValue<int32_t>();
		}
	}

	// Determine Lance path
	auto sanitized = SanitizeIndexName(info->index_name);
	auto &storage = table.GetStorage();
	auto &storage_manager = storage.db.GetStorageManager();
	auto db_path = storage_manager.GetDBPath();
	if (db_path.empty()) {
		state->lance_path = MakeUniqueTempPath(sanitized);
	} else {
		state->lance_path = db_path + ".lance/" + sanitized;
	}

	// Ensure parent directory exists
	auto &fs = FileSystem::GetFileSystem(context);
	auto parent = state->lance_path.substr(0, state->lance_path.rfind('/'));
	if (!parent.empty()) {
		fs.CreateDirectory(parent);
	}

	state->rust_handle = LanceCreateDetached(state->lance_path, state->dimension, state->metric, sanitized);
	return std::move(state);
}

SinkResultType PhysicalCreateLanceIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSinkInput &input) const {
	auto &state = input.global_state.Cast<CreateLanceGlobalSinkState>();

	auto col_count = chunk.ColumnCount();
	D_ASSERT(col_count >= 2);

	auto &vec_col = chunk.data[0];
	auto &rowid_col = chunk.data[col_count - 1];

	auto count = chunk.size();
	if (count == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &array_child = ArrayVector::GetEntry(vec_col);
	auto array_size = ArrayType::GetSize(vec_col.GetType());
	auto child_data = FlatVector::GetData<float>(array_child);

	UnifiedVectorFormat rowid_format;
	rowid_col.ToUnifiedFormat(count, rowid_format);
	auto rowid_data = reinterpret_cast<row_t *>(rowid_format.data);

	vector<int64_t> labels(count);
	auto n = LanceDetachedAddBatch(state.rust_handle, child_data,
	                               static_cast<int32_t>(count),
	                               state.dimension, labels.data());

	for (idx_t i = 0; i < static_cast<idx_t>(n); i++) {
		auto row_idx = rowid_format.sel->get_index(i);
		auto row_id = rowid_data[row_idx];
		state.label_to_rowid.push_back(row_id);
		state.rowid_to_label[row_id] = labels[i];
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalCreateLanceIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                    OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<CreateLanceGlobalSinkState>();

	auto &storage = table.GetStorage();
	if (!storage.IsMainTable()) {
		throw TransactionException("Transaction conflict: cannot add index to altered/dropped table");
	}

	case_insensitive_map_t<Value> options;
	options["metric"] = Value(state.metric);
	options["nprobes"] = Value::INTEGER(state.nprobes);
	options["refine_factor"] = Value::INTEGER(state.refine_factor);

	auto index = make_uniq<LanceIndex>(info->index_name, info->constraint_type, storage_ids,
	                                   TableIOManager::Get(storage), unbound_expressions, storage.db, options);

	index->rust_handle_ = state.rust_handle;
	state.rust_handle = nullptr;
	index->dimension_ = state.dimension;
	index->metric_ = state.metric;
	index->nprobes_ = state.nprobes;
	index->refine_factor_ = state.refine_factor;
	index->label_to_rowid_ = std::move(state.label_to_rowid);
	index->rowid_to_label_ = std::move(state.rowid_to_label);
	index->is_dirty_ = true;

	auto &schema = table.schema;
	info->column_ids = storage_ids;

	if (!alter_table_info) {
		auto entry = schema.GetEntry(schema.GetCatalogTransaction(context), CatalogType::INDEX_ENTRY, info->index_name);
		if (entry) {
			if (info->on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index '%s' already exists!", info->index_name);
			}
			return SinkFinalizeType::READY;
		}

		auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(context), *info, table).get();
		D_ASSERT(index_entry);
		auto &idx_entry = index_entry->Cast<DuckIndexEntry>();
		BoundIndex &bi = *index;
		idx_entry.initial_index_size = bi.GetInMemorySize();
	} else {
		auto &indexes = storage.GetDataTableInfo()->GetIndexes();
		indexes.Scan([&](Index &idx) {
			if (idx.GetIndexName() == info->index_name) {
				throw CatalogException("Index with name already exists: %s", info->index_name);
			}
			return false;
		});

		auto &catalog = Catalog::GetCatalog(context, info->catalog);
		catalog.Alter(context, *alter_table_info);
	}

	storage.AddIndex(std::move(index));
	return SinkFinalizeType::READY;
}

SourceResultType PhysicalCreateLanceIndex::GetData(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

} // namespace duckdb
