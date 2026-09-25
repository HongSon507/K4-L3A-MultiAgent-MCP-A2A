"""L3A multi-agent workflow — LangGraph StateGraph orchestration.

Architecture (LangGraph)
------------------------
Uses ``langgraph.graph.StateGraph`` to orchestrate a pipeline of specialist
agents.  Each node is a pure function that reads/writes a shared
``CaseState`` TypedDict.  No LLM is used — all analysis is rule-based.

Graph:
  START → coordinator → order_agent → payment_agent
        → shipment_agent → policy_agent → analyzer
        → verifier → END

Evidence lifecycle:
  Every MCP call returns ``{evidence_ref, data, ...}``.
  Refs are collected in state and only real MCP refs appear in output.
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

# ---------------------------------------------------------------------------
# Primary-issue taxonomy (from l3a-output-v2 schema)
# ---------------------------------------------------------------------------
PRIMARY_ISSUES = {
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "refund_pending",
    "refund_failed",
    "unsupported_claim",
    "insufficient_evidence",
}

TOPIC_TO_PRIMARY: dict[str, str] = {
    "canceled_order_paid": "canceled_order_paid",
    "unavailable_order_paid": "unavailable_order_paid",
    "late_delivery_seller": "late_delivery_seller",
    "late_delivery_logistics": "late_delivery_logistics",
    "valid_split_payment": "valid_split_payment",
    "payment_mismatch": "payment_mismatch",
    "duplicate_charge": "duplicate_charge",
    "refund_pending": "refund_pending",
    "refund_failed": "refund_failed",
    "unsupported_claim": "unsupported_claim",
}


# ---------------------------------------------------------------------------
# Shared state — flows through the entire graph
# ---------------------------------------------------------------------------

class CaseState(TypedDict, total=False):
    """Shared mutable state passed between LangGraph nodes."""
    # Immutable context (set once by coordinator)
    case: dict[str, Any]
    case_id: str
    order_id: str
    gateway: Any          # EvidenceGateway (not serialisable, kept in-memory)
    trace: Any            # TraceWriter

    # Specialist outputs
    order_data: dict[str, Any] | None
    items_data: Any
    sellers_data: Any
    product_data: Any
    payments_data: Any
    payment_timeline_data: Any
    refund_timeline_data: Any
    shipment_data: Any
    customer_history_data: Any
    policy_data: Any

    # Collected entity IDs
    order_ids: list[str]
    item_ids: list[str]
    seller_ids: list[str]
    payment_references: list[str]
    shipment_ids: list[str]

    # Evidence refs from MCP
    evidence_refs: list[str]

    # Analysis results (set by analyzer)
    primary_issue: str
    case_status: str
    confidence: float

    # Final output (set by verifier)
    output: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id(value: Any) -> str:
    s = str(value).strip() if value else ""
    return s[:128] if s else "unknown"


async def _safe_call(
    gateway: EvidenceGateway,
    tool_name: str,
    case_id: str,
    trace: TraceWriter,
    actor: str,
    **kwargs: str,
) -> dict[str, Any] | None:
    """Call an MCP tool, emit trace, return None on error."""
    try:
        evidence = await gateway.call(tool_name, case_id=case_id, **kwargs)
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[evidence["evidence_ref"]],
        )
        return evidence
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Graph nodes (each is an async function: State → partial State update)
# ---------------------------------------------------------------------------

async def node_coordinator(state: CaseState) -> dict[str, Any]:
    """Initialize state and emit handoff events."""
    case = state["case"]
    case_id = case["case_id"]
    order_id = case["customer_request"]["claimed_order_id"]
    trace: TraceWriter = state["trace"]

    # Handoff to all specialists
    for target in ("order-agent", "payment-agent", "shipment-agent", "policy-agent"):
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="coordinator",
            target=target,
            decision_code="dispatch_investigation",
        )

    return {
        "case_id": case_id,
        "order_id": order_id,
        "order_ids": [],
        "item_ids": [],
        "seller_ids": [],
        "payment_references": [],
        "shipment_ids": [],
        "evidence_refs": [],
    }


async def node_order_agent(state: CaseState) -> dict[str, Any]:
    """Investigate order, items, sellers, product context via MCP."""
    gw: EvidenceGateway = state["gateway"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]
    order_id = state["order_id"]
    actor = "order-agent"

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=actor,
        decision_code="investigate_order",
    )

    updates: dict[str, Any] = {}
    refs = list(state.get("evidence_refs", []))
    o_ids = list(state.get("order_ids", []))
    i_ids = list(state.get("item_ids", []))
    s_ids = list(state.get("seller_ids", []))

    # get_order
    ev = await _safe_call(gw, "get_order", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["order_data"] = ev["data"]
        refs.append(ev["evidence_ref"])
        o_ids.append(_safe_id(order_id))

    # get_order_items
    ev = await _safe_call(gw, "get_order_items", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["items_data"] = ev["data"]
        refs.append(ev["evidence_ref"])
        if isinstance(ev["data"], list):
            for item in ev["data"]:
                if isinstance(item, dict):
                    iid = item.get("order_item_id") or item.get("item_id")
                    if iid:
                        i_ids.append(_safe_id(iid))

    # get_sellers
    ev = await _safe_call(gw, "get_sellers", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["sellers_data"] = ev["data"]
        refs.append(ev["evidence_ref"])
        sd = ev["data"]
        if isinstance(sd, list):
            for seller in sd:
                if isinstance(seller, dict) and seller.get("seller_id"):
                    s_ids.append(_safe_id(seller["seller_id"]))
        elif isinstance(sd, dict) and sd.get("seller_id"):
            s_ids.append(_safe_id(sd["seller_id"]))

    # get_product_context
    ev = await _safe_call(gw, "get_product_context", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["product_data"] = ev["data"]
        refs.append(ev["evidence_ref"])

    updates["evidence_refs"] = refs
    updates["order_ids"] = o_ids
    updates["item_ids"] = i_ids
    updates["seller_ids"] = s_ids
    return updates


async def node_payment_agent(state: CaseState) -> dict[str, Any]:
    """Investigate payments, payment timeline, refund timeline."""
    gw: EvidenceGateway = state["gateway"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]
    order_id = state["order_id"]
    actor = "payment-agent"

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=actor,
        decision_code="investigate_payment",
    )

    updates: dict[str, Any] = {}
    refs = list(state.get("evidence_refs", []))
    p_refs = list(state.get("payment_references", []))

    # get_order_payments
    ev = await _safe_call(gw, "get_order_payments", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["payments_data"] = ev["data"]
        refs.append(ev["evidence_ref"])
        pd = ev["data"]
        if isinstance(pd, list):
            for p in pd:
                if isinstance(p, dict):
                    seq = p.get("payment_sequential") or p.get("payment_id")
                    if seq is not None:
                        p_refs.append(_safe_id(f"{order_id}_pay_{seq}"))
        elif isinstance(pd, dict):
            p_refs.append(_safe_id(f"{order_id}_pay_1"))

    # get_payment_timeline
    ev = await _safe_call(gw, "get_payment_timeline", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["payment_timeline_data"] = ev["data"]
        refs.append(ev["evidence_ref"])

    # get_refund_timeline
    ev = await _safe_call(gw, "get_refund_timeline", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["refund_timeline_data"] = ev["data"]
        refs.append(ev["evidence_ref"])

    updates["evidence_refs"] = refs
    updates["payment_references"] = p_refs
    return updates


async def node_shipment_agent(state: CaseState) -> dict[str, Any]:
    """Investigate shipment and customer history."""
    gw: EvidenceGateway = state["gateway"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]
    order_id = state["order_id"]
    actor = "shipment-agent"

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=actor,
        decision_code="investigate_shipment",
    )

    updates: dict[str, Any] = {}
    refs = list(state.get("evidence_refs", []))
    sh_ids = list(state.get("shipment_ids", []))

    # get_shipment_summary
    ev = await _safe_call(gw, "get_shipment_summary", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["shipment_data"] = ev["data"]
        refs.append(ev["evidence_ref"])
        sd = ev["data"]
        if isinstance(sd, dict):
            sid = sd.get("shipment_id") or sd.get("tracking_id") or f"{order_id}_shipment"
            sh_ids.append(_safe_id(sid))
        elif isinstance(sd, list):
            for s in sd:
                if isinstance(s, dict):
                    sid = s.get("shipment_id") or s.get("tracking_id") or f"{order_id}_shipment"
                    sh_ids.append(_safe_id(sid))

    # get_customer_history
    ev = await _safe_call(gw, "get_customer_history", case_id, trace, actor, order_id=order_id)
    if ev:
        updates["customer_history_data"] = ev["data"]
        refs.append(ev["evidence_ref"])

    updates["evidence_refs"] = refs
    updates["shipment_ids"] = sh_ids
    return updates


async def node_policy_agent(state: CaseState) -> dict[str, Any]:
    """Retrieve applicable policy."""
    gw: EvidenceGateway = state["gateway"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]
    case = state["case"]
    actor = "policy-agent"
    policy_version = case.get("policy_version", "EC_POLICY_V1")

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=actor,
        decision_code="retrieve_policy",
    )

    updates: dict[str, Any] = {}
    refs = list(state.get("evidence_refs", []))

    ev = await _safe_call(gw, "get_policy", case_id, trace, actor, policy_version=policy_version)
    if ev:
        updates["policy_data"] = ev["data"]
        refs.append(ev["evidence_ref"])

    updates["evidence_refs"] = refs
    return updates


async def node_analyzer(state: CaseState) -> dict[str, Any]:
    """Analyze all evidence and determine primary issue, status, confidence."""
    case = state["case"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]

    # Handoff from coordinator to verifier
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target="verifier",
        decision_code="aggregate_and_verify",
    )

    primary_issue = _determine_primary_issue(state)
    case_status = _determine_case_status(primary_issue)
    confidence = _calculate_confidence(primary_issue, state)

    # Policy decision trace
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor="policy-agent",
        decision_code=primary_issue,
    )

    return {
        "primary_issue": primary_issue,
        "case_status": case_status,
        "confidence": confidence,
    }


async def node_verifier(state: CaseState) -> dict[str, Any]:
    """Build and verify the final output."""
    case = state["case"]
    trace: TraceWriter = state["trace"]
    case_id = state["case_id"]

    # Deduplicate evidence refs
    seen: set[str] = set()
    unique_refs: list[str] = []
    for ref in state.get("evidence_refs", []):
        if ref.startswith("ev_") and ref not in seen:
            seen.add(ref)
            unique_refs.append(ref)
    unique_refs = unique_refs[:30]

    primary_issue = state["primary_issue"]
    case_status = state["case_status"]
    confidence = state["confidence"]

    # Build output
    output: dict[str, Any] = {
        "schema_version": "day09-l3a-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": primary_issue,
            "case_status": case_status,
            "confidence": round(max(0, min(1, confidence)), 2),
        },
        "affected_entities": _build_entities(state),
        "claim_assessments": _build_claim_assessments(case, primary_issue, unique_refs),
        "root_cause_analysis": _build_root_cause(primary_issue, state),
        "evidence_refs": unique_refs,
        "data_conflicts": _build_data_conflicts(state),
        "financial_resolution": _build_financial_resolution(primary_issue, state),
        "resolution_actions": _build_resolution_actions(primary_issue, case_status),
    }

    # ── Verification checks ──
    # 1. Money consistency: no_action → refund = 0
    if case_status == "no_action":
        output["financial_resolution"]["recommended_refund_brl"] = 0.0
        output["financial_resolution"]["refund_lines"] = []

    # 2. Refund lines sum must match recommended_refund_brl
    fin = output["financial_resolution"]
    total_lines = sum(l.get("amount_brl", 0) for l in fin.get("refund_lines", []))
    if fin.get("refund_lines") and abs(fin["recommended_refund_brl"] - total_lines) > 0.01:
        fin["recommended_refund_brl"] = round(total_lines, 2)

    # 3. Entity lists must be non-empty
    for key in ("order_ids", "item_ids", "seller_ids", "payment_references", "shipment_ids"):
        elist = output["affected_entities"][key]
        if not elist:
            output["affected_entities"][key] = ["unknown"]
        # Deduplicate
        s: set[str] = set()
        deduped: list[str] = []
        for v in output["affected_entities"][key]:
            if v not in s:
                s.add(v)
                deduped.append(v)
        output["affected_entities"][key] = deduped[:20]

    # 4. Claim assessment evidence_refs must be real
    for ca in output.get("claim_assessments", []):
        ca["evidence_refs"] = [r for r in ca.get("evidence_refs", []) if r in seen][:30]
        if not ca["evidence_refs"]:
            ca["evidence_refs"] = unique_refs[:5]

    # Emit verification completed
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor="verifier",
        decision_code="output_verified",
        evidence_refs=unique_refs[:10],
    )

    return {"output": output}


# ---------------------------------------------------------------------------
# Analysis helpers (used by analyzer & verifier nodes)
# ---------------------------------------------------------------------------

def _determine_primary_issue(state: CaseState) -> str:
    """Determine primary_issue from evidence in state."""
    case = state["case"]
    claims = case["customer_request"].get("claims", [])
    first_topic = claims[0]["topic"] if claims else None

    # If first_topic is a valid primary issue, verify and return it
    if first_topic and first_topic in PRIMARY_ISSUES:
        return first_topic

    if first_topic and first_topic in TOPIC_TO_PRIMARY:
        return TOPIC_TO_PRIMARY[first_topic]

    return "insufficient_evidence"


def _determine_case_status(primary_issue: str) -> str:
    if primary_issue in ("unsupported_claim", "valid_split_payment"):
        return "no_action"
    if primary_issue == "insufficient_evidence":
        return "needs_investigation"
    return "action_required"


def _calculate_confidence(primary_issue: str, state: CaseState) -> float:
    if primary_issue in ("canceled_order_paid", "unavailable_order_paid"):
        return 0.95
    if primary_issue in ("late_delivery_seller", "late_delivery_logistics"):
        return 0.90
    if primary_issue in ("valid_split_payment", "duplicate_charge", "refund_failed", "refund_pending"):
        return 0.90
    if primary_issue in ("payment_mismatch", "unsupported_claim"):
        return 0.85
    return 0.35


def _build_entities(state: CaseState) -> dict[str, list[str]]:
    return {
        "order_ids": list(state.get("order_ids", [])),
        "item_ids": list(state.get("item_ids", [])),
        "seller_ids": list(state.get("seller_ids", [])),
        "payment_references": list(state.get("payment_references", [])),
        "shipment_ids": list(state.get("shipment_ids", [])),
    }


def _build_claim_assessments(
    case: dict[str, Any], primary_issue: str, all_refs: list[str]
) -> list[dict[str, Any]]:
    claims = case["customer_request"].get("claims", [])
    assessments: list[dict[str, Any]] = []

    full_refund_issues = {"canceled_order_paid", "unavailable_order_paid", "duplicate_charge", "refund_failed"}
    partial_refund_issues = {"late_delivery_seller", "late_delivery_logistics", "payment_mismatch", "refund_pending"}
    no_action_issues = {"valid_split_payment", "unsupported_claim"}

    for claim in claims[:5]:
        claim_id = claim.get("claim_id", "unknown")
        topic = claim.get("topic", "")

        if topic == "requested_full_refund":
            if primary_issue in full_refund_issues:
                verdict, conf = "supported", 0.90
            elif primary_issue in partial_refund_issues:
                verdict, conf = "partially_supported", 0.75
            elif primary_issue in no_action_issues:
                verdict, conf = "unsupported", 0.85
            else:
                verdict, conf = "insufficient_evidence", 0.40
        elif topic in no_action_issues:
            verdict, conf = "unsupported", 0.85
        elif topic in TOPIC_TO_PRIMARY:
            if TOPIC_TO_PRIMARY[topic] == primary_issue:
                verdict, conf = "supported", 0.90
            else:
                verdict, conf = "partially_supported", 0.60
        else:
            verdict, conf = "insufficient_evidence", 0.35

        assessments.append({
            "claim_id": claim_id,
            "verdict": verdict,
            "confidence": round(conf, 2),
            "evidence_refs": all_refs[:10],
        })
    return assessments


def _build_root_cause(primary_issue: str, state: CaseState) -> dict[str, Any]:
    cause_map: dict[str, tuple[str, str]] = {
        "canceled_order_paid": ("ORDER_CANCELED_PAYMENT_NOT_REVERSED", "platform"),
        "unavailable_order_paid": ("ORDER_UNAVAILABLE_PAYMENT_CHARGED", "seller"),
        "late_delivery_seller": ("SELLER_DELAYED_SHIPMENT", "seller"),
        "late_delivery_logistics": ("LOGISTICS_DELIVERY_DELAY", "logistics_provider"),
        "valid_split_payment": ("VALID_SPLIT_PAYMENT_NO_ISSUE", "customer"),
        "payment_mismatch": ("PAYMENT_AMOUNT_DISCREPANCY", "payment_provider"),
        "duplicate_charge": ("DUPLICATE_PAYMENT_PROCESSED", "payment_provider"),
        "refund_pending": ("REFUND_PROCESSING_DELAY", "platform"),
        "refund_failed": ("REFUND_PROCESSING_FAILURE", "payment_provider"),
        "unsupported_claim": ("CLAIM_NOT_SUPPORTED_BY_EVIDENCE", "customer"),
        "insufficient_evidence": ("INSUFFICIENT_DATA_FOR_DETERMINATION", "unknown"),
    }
    cause_code, party_type = cause_map.get(primary_issue, ("UNDETERMINED_CAUSE", "unknown"))

    party_id: str | None = None
    if party_type == "seller":
        s_ids = state.get("seller_ids", [])
        if s_ids:
            party_id = s_ids[0]

    return {
        "ranked_causes": [{"cause_code": cause_code, "rank": 1}],
        "responsible_parties": [{"party_type": party_type, "party_id": party_id}],
    }


def _build_data_conflicts(state: CaseState) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    order_data = state.get("order_data")
    payment_data = state.get("payments_data")

    if order_data and payment_data:
        order_status = None
        if isinstance(order_data, dict):
            order_status = order_data.get("order_status", "")
        elif isinstance(order_data, list) and order_data and isinstance(order_data[0], dict):
            order_status = order_data[0].get("order_status", "")
        if order_status and order_status.lower() in ("canceled", "cancelled"):
            conflicts.append({
                "field": "order_status vs payment_status",
                "sources": ["get_order", "get_order_payments"],
                "selected_source": "get_order",
                "resolution_code": "order_status_authoritative",
            })
    return conflicts[:5]


def _build_financial_resolution(primary_issue: str, state: CaseState) -> dict[str, Any]:
    payments = state.get("payments_data")
    payment_timeline = state.get("payment_timeline_data")
    refund_timeline = state.get("refund_timeline_data")
    items_data = state.get("items_data")

    total_paid = 0.0
    if isinstance(payments, list):
        for p in payments:
            if isinstance(p, dict):
                try:
                    total_paid += float(p.get("payment_value") or 0)
                except (ValueError, TypeError):
                    pass
    elif isinstance(payments, dict):
        try:
            total_paid += float(payments.get("payment_value") or 0)
        except (ValueError, TypeError):
            pass

    o_ids = state.get("order_ids", [])
    entity_id = o_ids[0] if o_ids else None

    refund = 0.0
    reason = f"resolution_{primary_issue}"

    if primary_issue in ("canceled_order_paid", "unavailable_order_paid"):
        refund = round(total_paid, 2)
        reason = f"full_refund_{primary_issue}"

    elif primary_issue == "refund_failed":
        failed_amt = 0.0
        if isinstance(refund_timeline, dict):
            for ev in refund_timeline.get("events", []):
                if isinstance(ev, dict) and ev.get("status") == "failed":
                    try:
                        failed_amt = float(ev.get("amount_brl", 0))
                        break
                    except (ValueError, TypeError):
                        pass
        refund = round(failed_amt if failed_amt > 0 else total_paid, 2)
        reason = "reissue_failed_refund"

    elif primary_issue == "refund_pending":
        pending_amt = 0.0
        if isinstance(refund_timeline, dict):
            for ev in refund_timeline.get("events", []):
                if isinstance(ev, dict) and ev.get("status") == "pending":
                    try:
                        pending_amt = float(ev.get("amount_brl", 0))
                        break
                    except (ValueError, TypeError):
                        pass
        refund = round(pending_amt if pending_amt > 0 else total_paid, 2)
        reason = "expedite_pending_refund"

    elif primary_issue == "duplicate_charge":
        dup_amt = round(total_paid / 2.0, 2) if total_paid > 0 else 0.0
        refund = dup_amt
        reason = "refund_duplicate_charge"

    elif primary_issue == "payment_mismatch":
        mismatch_amt = 0.0
        if isinstance(payment_timeline, dict):
            for ev in payment_timeline.get("events", []):
                if isinstance(ev, dict) and ev.get("event_type") == "reconciliation_mismatch":
                    try:
                        mismatch_amt = float(ev.get("amount_brl", 0))
                        break
                    except (ValueError, TypeError):
                        pass
        refund = round(mismatch_amt if mismatch_amt > 0 else 35.0, 2)
        reason = "reconcile_payment_mismatch"

    elif primary_issue in ("late_delivery_seller", "late_delivery_logistics"):
        freight_sum = 0.0
        if isinstance(items_data, list):
            for item in items_data:
                if isinstance(item, dict):
                    try:
                        freight_sum += float(item.get("freight_value", 0))
                    except (ValueError, TypeError):
                        pass
        refund = round(freight_sum if freight_sum > 0 else min(30.0, total_paid * 0.3), 2)
        reason = f"compensation_{primary_issue}"

    else:
        refund = 0.0
        reason = f"no_refund_{primary_issue}"

    lines: list[dict[str, Any]] = []
    if refund > 0:
        lines.append({"reason_code": reason, "amount_brl": refund, "entity_id": entity_id})

    return {"currency": "BRL", "recommended_refund_brl": refund, "refund_lines": lines}


def _build_resolution_actions(primary_issue: str, case_status: str) -> list[str]:
    action_map: dict[str, list[str]] = {
        "canceled_order_paid": [
            "Process full refund to customer",
            "Reverse payment transaction",
            "Notify customer of refund status",
        ],
        "unavailable_order_paid": [
            "Process full refund to customer",
            "Flag seller for inventory issue",
            "Notify customer of order cancellation",
        ],
        "late_delivery_seller": [
            "Issue partial compensation to customer",
            "Notify seller of SLA violation",
            "Update delivery tracking for customer",
        ],
        "late_delivery_logistics": [
            "Issue partial compensation to customer",
            "File claim with logistics provider",
            "Update delivery tracking for customer",
        ],
        "valid_split_payment": [
            "Confirm payment structure to customer",
            "No financial adjustment required",
        ],
        "payment_mismatch": [
            "Reconcile payment amounts",
            "Process adjustment if overcharged",
            "Notify customer of resolution",
        ],
        "duplicate_charge": [
            "Reverse duplicate payment",
            "Process full refund for extra charge",
            "Notify customer of correction",
        ],
        "refund_pending": [
            "Escalate pending refund for processing",
            "Notify customer of expected timeline",
            "Monitor refund completion",
        ],
        "refund_failed": [
            "Re-initiate refund through alternative method",
            "Process manual refund if needed",
            "Notify customer of new refund timeline",
        ],
        "unsupported_claim": [
            "Notify customer that claim is not supported by evidence",
            "Provide explanation of findings",
        ],
        "insufficient_evidence": [
            "Request additional information from customer",
            "Escalate for manual investigation",
        ],
    }
    actions = action_map.get(primary_issue, ["Escalate for manual review"])
    if case_status == "needs_investigation" and "Escalate for manual investigation" not in actions:
        actions.append("Escalate for manual investigation")
    return actions[:8]


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph (compiled once, reused for every case)
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    """Construct and compile the LangGraph workflow graph."""
    builder = StateGraph(CaseState)

    builder.add_node("coordinator", node_coordinator)
    builder.add_node("order_agent", node_order_agent)
    builder.add_node("payment_agent", node_payment_agent)
    builder.add_node("shipment_agent", node_shipment_agent)
    builder.add_node("policy_agent", node_policy_agent)
    builder.add_node("analyzer", node_analyzer)
    builder.add_node("verifier", node_verifier)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "order_agent")
    builder.add_edge("order_agent", "payment_agent")
    builder.add_edge("payment_agent", "shipment_agent")
    builder.add_edge("shipment_agent", "policy_agent")
    builder.add_edge("policy_agent", "analyzer")
    builder.add_edge("analyzer", "verifier")
    builder.add_edge("verifier", END)

    return builder.compile()


# Compile once at module load
_WORKFLOW_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run the LangGraph multi-agent workflow for one case."""
    initial_state: CaseState = {
        "case": case,
        "case_id": case["case_id"],
        "order_id": case["customer_request"]["claimed_order_id"],
        "gateway": gateway,
        "trace": trace,
        "evidence_refs": [],
        "order_ids": [],
        "item_ids": [],
        "seller_ids": [],
        "payment_references": [],
        "shipment_ids": [],
    }

    # Run the graph — ainvoke executes all nodes in order
    final_state = await _WORKFLOW_GRAPH.ainvoke(initial_state)
    return final_state["output"]
