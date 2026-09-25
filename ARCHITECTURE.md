# L3A Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Hệ thống multi-agent sử dụng **LangGraph StateGraph** để điều phối pipeline điều tra khiếu nại TMĐT. Mỗi node trong graph là một specialist agent (async function) đọc/ghi shared `CaseState` TypedDict. Không sử dụng LLM — toàn bộ logic phân tích là rule-based.

```text
┌─────────────────── LangGraph StateGraph ───────────────────┐
│                                                             │
│  START → coordinator → order_agent → payment_agent          │
│        → shipment_agent → policy_agent → analyzer           │
│        → verifier → END                                     │
│                                                             │
│  Shared state: CaseState (TypedDict)                        │
│  Evidence refs: collected across all nodes                  │
│  MCP calls: 10 tools via EvidenceGateway                    │
└─────────────────────────────────────────────────────────────┘
```

**Framework**: `langgraph==1.2.12`

## 2. Agent ownership

| Actor (Graph Node) | Input                    | Trách nhiệm                                                                               | Output/handoff                                  |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| coordinator         | case JSON                | Khởi tạo CaseState, emit handoff events tới 4 specialists                                  | case_id, order_id, empty entity lists           |
| order_agent         | CaseState                | Gọi get_order, get_order_items, get_sellers, get_product_context. Thu thập entity IDs      | order_data, item_ids, seller_ids, evidence_refs |
| payment_agent       | CaseState                | Gọi get_order_payments, get_payment_timeline, get_refund_timeline. Phân tích thanh toán    | payments_data, payment_references, evidence_refs|
| shipment_agent      | CaseState                | Gọi get_shipment_summary, get_customer_history. Phân tích vận chuyển                      | shipment_data, shipment_ids, evidence_refs      |
| policy_agent        | CaseState                | Gọi get_policy. Lấy chính sách áp dụng cho case                                           | policy_data, evidence_refs                      |
| analyzer            | CaseState (full)         | Phân tích evidence, xác định primary_issue, case_status, confidence                       | primary_issue, case_status, confidence          |
| verifier            | CaseState (full)         | Cross-check, build final output, validate consistency, emit verification_completed          | output (final JSON)                             |

Tool ownership (mỗi agent chỉ gọi tool được phân quyền):

| Tool                  | Graph Node (Agent) |
| --------------------- | -------------------- |
| get_order             | order_agent          |
| get_order_items       | order_agent          |
| get_sellers           | order_agent          |
| get_product_context   | order_agent          |
| get_order_payments    | payment_agent        |
| get_payment_timeline  | payment_agent        |
| get_refund_timeline   | payment_agent        |
| get_shipment_summary  | shipment_agent       |
| get_customer_history  | shipment_agent       |
| get_policy            | policy_agent         |

## 3. A2A protocol

- **Message envelope**: LangGraph `CaseState` TypedDict là shared state — mỗi node return partial update dict được merge vào state.
- **Correlation**: Mọi MCP call và trace event đều gắn `case_id` từ input case. Không dùng evidence chéo case.
- **Handoff**: Coordinator emit `handoff` event trước mỗi specialist. Specialist emit `task_assigned` khi bắt đầu, `tool_result_consumed` cho mỗi MCP response.
- **Graph flow**: `START → coordinator → order_agent → payment_agent → shipment_agent → policy_agent → analyzer → verifier → END`. Tuyến tính, không có điều kiện rẽ nhánh hay vòng lặp.
- **Timeout**: MCP gateway timeout 300s (connect 30s). Nếu tool fail, node set field = None và tiếp tục.

## 4. Evidence lifecycle

1. Mỗi MCP call trả về envelope `{evidence_ref, data, result_hash, domain}`.
2. `evidence_ref` được validate bởi `Contracts.validate_evidence()` — phải match pattern `^ev_[A-Za-z0-9_-]{20,96}$`.
3. Sau khi nhận evidence, node emit `tool_result_consumed` trace event với `evidence_refs` array.
4. Evidence refs được accumulate trong `CaseState.evidence_refs` qua các nodes.
5. Verifier deduplicate refs, gắn vào output `evidence_refs` (max 30, unique) và `claim_assessments`.
6. **Không tự tạo evidence_ref**. Chỉ dùng ref từ MCP response.
7. **Không tái sử dụng evidence giữa các case**.

## 5. Failure policy

| Failure                    | Retry? | Fallback                                | Trace event/code           |
| -------------------------- | ------ | --------------------------------------- | -------------------------- |
| MCP timeout                | No     | Node set field = None, graph tiếp tục   | Không emit tool_result     |
| Not found                  | No     | Field = None, entity_ids fallback ["unknown"]| Không emit tool_result |
| Source conflict             | No     | Ưu tiên order data (authoritative)      | data_conflicts in output   |
| Invalid specialist result  | No     | Verifier node sửa/fallback giá trị hợp lệ | verification_completed  |

Retry không được implement vì MCP calls đã được audit — retry có thể gây duplicate audit entries.

## 6. Verification invariants

Verifier node kiểm tra trước finalize:

1. **Schema**: `schema_version` = `"day09-l3a-output-v2"`, `case_id` khớp input.
2. **Entity scope**: Tất cả entity IDs trong `affected_entities` phải non-empty (fallback `["unknown"]`), deduplicated, max 20.
3. **Evidence ownership**: Chỉ evidence_refs bắt đầu bằng `ev_` và thực sự từ MCP mới được include.
4. **Claim linkage**: Mỗi `claim_assessment` phải có ít nhất 1 evidence_ref thật.
5. **Money totals**: `recommended_refund_brl` = sum of `refund_lines[].amount_brl`.
6. **Responsibility/action consistency**: `no_action` → refund = 0, refund_lines = [].
7. **Confidence bounds**: 0 ≤ confidence ≤ 1, calibrated theo available evidence.
8. **Deduplication**: evidence_refs và entity IDs unique.

## 7. Reproducibility

- **Framework**: `langgraph==1.2.12`
- **Dependencies**: `httpx2>=2,<3`, `mcp>=2,<3`, `jsonschema[format]>=4.25,<5`, `python-dotenv>=1.1,<2`
- **Python**: ≥ 3.11
- **Concurrency**: Sequential (1 case at a time, nodes executed in graph order)
- **Random seed**: `secrets.token_urlsafe(18)` cho event_id — non-deterministic by design
- **Lệnh chạy**:
  ```bash
  pip install langgraph
  python -m pip install -e ".[dev]"
  day09 run
  day09 validate
  day09 package --output dist/submission.zip
  ```
- **Giới hạn**: 10 MCP calls per case (all 10 tools), timeout 300s per call.
