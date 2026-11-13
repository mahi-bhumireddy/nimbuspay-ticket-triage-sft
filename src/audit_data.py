import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

VALID_CATEGORIES = {
    "payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"
}
VALID_CHANNELS = {"upi", "card", "netbanking", "wallet"}
VALID_LANGUAGES = {"en", "hi-en"}
NO_TXN_CATEGORIES = {"account_access", "kyc", "other"}
EXPECTED_KEYS = [
    "category", "priority", "amount_paise", "txn_date",
    "txn_id", "channel", "language", "needs_human"
]

LEGAL_PATTERN = re.compile(
    r"\b(rbi|ombudsman|consumer\s*court|lawyer|legal\s*notice|legal\s*action)\b",
    re.IGNORECASE,
)

PRIOR_CONTACT_PATTERN = re.compile(
    r"\b(third\s*time|second\s*time|2nd\s*time|3rd\s*time|multiple\s*times|again\s*and\s*again|"
    r"baar\s*baar|bar\s*bar|already\s*(contacted|raised|complained|emailed|messaged|called|reached\s*out)|"
    r"earlier\s*ticket|previous\s*ticket|last\s*ticket|follow\s*up|following\s*up)\b",
    re.IGNORECASE,
)


def parse_received_date(raw_text: str):
    m = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", raw_text)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date()
    return None


def strip_quoted_lines(raw_text: str) -> str:
    return "\n".join(line for line in raw_text.splitlines() if not line.lstrip().startswith(">"))


def normalize_txn_id(val):
    if val is None or not isinstance(val, str):
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", val).upper()
    if re.fullmatch(r"NP\d{10}", cleaned):
        return cleaned
    return None


def coerce_int(val):
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "")
        if re.fullmatch(r"\d+", s):
            return int(s)
        if re.fullmatch(r"\d+\.\d+", s):
            return int(round(float(s) * 100))
    return "INVALID"


def coerce_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.strip().lower() == "true":
            return True
        if val.strip().lower() == "false":
            return False
    if val in (0, 1):
        return bool(val)
    return "INVALID"


def audit_and_clean(
    train_path="data/train.jsonl",
    clean_path="data/train_clean.jsonl",
    log_path="cleaning_log.csv",
):
    issue_counts = Counter()
    seen_ids = set()
    seen_texts = set()
    cleaned_rows = []
    log_rows = []
    total = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total += 1
            line = line.strip()
            if not line:
                issue_counts["empty_line"] += 1
                log_rows.append({"id": f"line-{line_no}", "action": "drop", "reason": "empty_line"})
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                issue_counts["corrupt_json"] += 1
                log_rows.append({"id": f"line-{line_no}", "action": "drop", "reason": "corrupt_json"})
                continue

            rid = rec.get("id", f"line-{line_no}")
            messages = rec.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                issue_counts["malformed_messages"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": "malformed_messages"})
                continue

            user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            asst_msg = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), "")

            if not user_msg or not asst_msg:
                issue_counts["missing_user_or_assistant"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": "missing_user_or_assistant"})
                continue

            try:
                target = json.loads(asst_msg)
            except json.JSONDecodeError:
                issue_counts["corrupt_assistant_json"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": "corrupt_assistant_json"})
                continue

            if not isinstance(target, dict):
                issue_counts["non_dict_target"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": "non_dict_target"})
                continue

            # Duplicate check
            if rid in seen_ids or user_msg.strip() in seen_texts:
                issue_counts["duplicate_row"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": "duplicate_row"})
                continue
            seen_ids.add(rid)
            seen_texts.add(user_msg.strip())

            reasons = []
            unquoted_text = strip_quoted_lines(user_msg)
            rec_date = parse_received_date(user_msg)

            # Check missing / extra keys
            missing_keys = set(EXPECTED_KEYS) - set(target.keys())
            if missing_keys:
                issue_counts["missing_required_keys"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": f"missing_keys:{sorted(missing_keys)}"})
                continue

            extra_keys = set(target.keys()) - set(EXPECTED_KEYS)
            if extra_keys:
                reasons.append(f"removed_extra_keys:{sorted(extra_keys)}")
                issue_counts["extra_keys"] += 1

            # Category
            cat = str(target.get("category", "")).strip().lower()
            if cat not in VALID_CATEGORIES:
                issue_counts["invalid_category"] += 1
                log_rows.append({"id": rid, "action": "drop", "reason": f"invalid_category:{cat}"})
                continue
            if cat != target.get("category"):
                reasons.append("normalized_category_casing")
                issue_counts["normalized_category_casing"] += 1

            # Language
            lang = str(target.get("language", "")).strip().lower()
            if lang not in VALID_LANGUAGES:
                reasons.append(f"invalid_language:{lang}->en")
                issue_counts["invalid_language"] += 1
                lang = "en"

            # amount_paise (type coercion + rupees-to-paise unit fix)
            amt = coerce_int(target.get("amount_paise"))
            if amt == "INVALID":
                reasons.append(f"invalid_amount:{target.get('amount_paise')}->None")
                issue_counts["invalid_amount"] += 1
                amt = None
            elif amt != target.get("amount_paise") or type(amt) is not type(target.get("amount_paise")):
                reasons.append("coerced_amount_to_int")
                issue_counts["coerced_amount_to_int"] += 1

            if isinstance(amt, int) and amt > 0:
                rupee_pat = re.compile(
                    rf"(?:\u20b9|rs\.?|inr)\s*{amt}\b|\b{amt}\s*(?:rupees|rs\.?|inr)\b",
                    re.IGNORECASE,
                )
                if rupee_pat.search(unquoted_text):
                    reasons.append(f"rupees_to_paise:{amt}->{amt * 100}")
                    issue_counts["rupees_to_paise"] += 1
                    amt = amt * 100

            # txn_date
            txn_date = target.get("txn_date")
            if txn_date is not None:
                if not isinstance(txn_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", txn_date):
                    reasons.append(f"malformed_txn_date:{txn_date}->None")
                    issue_counts["malformed_txn_date"] += 1
                    txn_date = None
                elif rec_date:
                    try:
                        parsed_td = datetime.strptime(txn_date, "%Y-%m-%d").date()
                        if parsed_td > rec_date:
                            reasons.append(f"future_txn_date:{txn_date}>received:{rec_date}->None")
                            issue_counts["future_txn_date"] += 1
                            txn_date = None
                    except ValueError:
                        reasons.append(f"invalid_calendar_date:{txn_date}->None")
                        issue_counts["invalid_calendar_date"] += 1
                        txn_date = None

            # txn_id
            raw_tid = target.get("txn_id")
            norm_tid = normalize_txn_id(raw_tid)
            if raw_tid != norm_tid:
                reasons.append(f"txn_id_fix:{raw_tid}->{norm_tid}")
                issue_counts["txn_id_fix"] += 1

            # channel
            chan = target.get("channel")
            if isinstance(chan, str):
                chan_clean = chan.strip().lower()
                chan = chan_clean if chan_clean in VALID_CHANNELS else None
            elif chan is not None:
                chan = None
            if chan != target.get("channel"):
                reasons.append(f"channel_fix:{target.get('channel')}->{chan}")
                issue_counts["channel_fix"] += 1

            # Null transaction fields for account_access, kyc, other
            if cat in NO_TXN_CATEGORIES:
                if any(x is not None for x in (amt, txn_date, norm_tid, chan)):
                    reasons.append(f"nulled_txn_fields_for_{cat}")
                    issue_counts["nulled_txn_fields_for_non_txn_cat"] += 1
                    amt, txn_date, norm_tid, chan = None, None, None, None

            # Priority derivation (SCHEMA.md v2)
            has_legal = bool(LEGAL_PATTERN.search(unquoted_text))
            if cat == "fraud" or (amt is not None and amt >= 5000000) or has_legal:
                expected_prio = "P1"
            elif (cat in {"payment_failed", "refund"} and amt is not None and amt >= 500000) or cat == "account_access":
                expected_prio = "P2"
            else:
                expected_prio = "P3"

            orig_prio = target.get("priority")
            if orig_prio != expected_prio:
                reasons.append(f"priority_fix:{orig_prio}->{expected_prio}")
                issue_counts["priority_fix"] += 1

            # needs_human derivation (SCHEMA.md v2)
            orig_nh = coerce_bool(target.get("needs_human"))
            if orig_nh == "INVALID":
                orig_nh = False
                reasons.append("coerced_needs_human_bool")
                issue_counts["coerced_needs_human_bool"] += 1
            elif type(orig_nh) is not type(target.get("needs_human")):
                reasons.append("coerced_needs_human_bool")
                issue_counts["coerced_needs_human_bool"] += 1

            has_prior = bool(PRIOR_CONTACT_PATTERN.search(unquoted_text)) or ("\n>" in user_msg)
            expected_nh = True if (expected_prio == "P1" or has_prior or orig_nh) else False
            if target.get("needs_human") != expected_nh:
                reasons.append(f"needs_human_fix:{target.get('needs_human')}->{expected_nh}")
                issue_counts["needs_human_fix"] += 1

            clean_obj = {
                "category": cat,
                "priority": expected_prio,
                "amount_paise": amt,
                "txn_date": txn_date,
                "txn_id": norm_tid,
                "channel": chan,
                "language": lang,
                "needs_human": expected_nh,
            }

            new_messages = []
            for m in messages:
                if m.get("role") == "assistant":
                    new_messages.append({"role": "assistant", "content": json.dumps(clean_obj, ensure_ascii=False)})
                else:
                    new_messages.append(m)

            cleaned_rows.append({"id": rid, "messages": new_messages})
            if reasons:
                log_rows.append({"id": rid, "action": "fix", "reason": "; ".join(reasons)})

    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    with open(clean_path, "w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "action", "reason"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Total rows read: {total}")
    print(f"Clean rows written: {len(cleaned_rows)}")
    print(f"Dropped rows: {sum(1 for r in log_rows if r['action'] == 'drop')}")
    print(f"Fixed rows: {sum(1 for r in log_rows if r['action'] == 'fix')}")
    print(f"Unchanged rows: {len(cleaned_rows) - sum(1 for r in log_rows if r['action'] == 'fix')}")
    print("\nBreakdown of issues detected:")
    for k, v in issue_counts.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    audit_and_clean()