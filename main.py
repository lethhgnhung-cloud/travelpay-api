"""
=============================================================
TravelPay Vietnam — Custom OpenAPI Backend
=============================================================
Endpoints:
  GET  /                        Health check
  GET  /exchange-rates          Tỷ giá live từ Vietcombank XML
  POST /transfer                Giao dịch chuyển tiền VND
  GET  /balance/{user_id}       Số dư tài khoản VND
  GET  /transactions/{user_id}  Lịch sử giao dịch
  GET  /scam-risk               AI scam risk score toàn vùng
  GET  /scam-risk/location      Scam risk theo địa điểm cụ thể
  GET  /scam-alerts             Top cảnh báo scam mới nhất
  GET  /docs                    Swagger UI (OpenAPI spec tự động)
  GET  /openapi.json            File spec để nộp deliverable

Tech: FastAPI + httpx + pandas
Run:  uvicorn main:app --reload --port 8000
=============================================================
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import httpx
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import uuid
import json
from datetime import datetime, timedelta
import random

# ─── APP SETUP ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TravelPay Vietnam API",
    description="""
## Custom OpenAPI Backend — TravelPay Vietnam

API hỗ trợ chuyển tiền VND cho khách du lịch tại Việt Nam với tích hợp 
**AI Scam Risk Intelligence** từ phân tích 321 review Reddit thực tế.

### Tính năng chính
- **Tỷ giá realtime** từ Vietcombank (thay thế Stripe/Plaid bị giới hạn VND)
- **Giao dịch VND** với validation và mock processing
- **AI Scam Risk Score** — cảnh báo khu vực nguy hiểm dựa trên Alternative Data
- **Scam Alerts** — top cảnh báo theo loại scam và địa điểm

### Alternative Data (Part A)
Scam risk được tính từ 321 Reddit reviews (r/vietnam, r/travel, r/solotravel)  
đã qua Ensemble Sentiment Model (VADER 70% + TextBlob 30%).
    """,
    version="1.0.0",
    contact={"name": "TravelPay Vietnam Team"},
    license_info={"name": "MIT"},
)

# Allow requests từ Google AI Studio và frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── LOAD AI SCAM DATA ────────────────────────────────────────────────────────
try:
    df = pd.read_csv("vietnam_scam_sentiment_results.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    SCAM_DATA_LOADED = True
    print(f"✅ Loaded {len(df)} scam reviews")
except Exception as e:
    print(f"⚠️  Could not load CSV: {e}. Using fallback data.")
    SCAM_DATA_LOADED = False
    df = pd.DataFrame()

SCAM_TYPES = [
    "Fake Grab / Taxi", "Overcharging", "Currency / Change",
    "Theft / Pickpocket", "Fake Products", "Phone / Digital",
    "Accommodation", "Tour / Activity",
]

# Vietnam locations + subreddit keyword mapping
LOCATION_KEYWORDS = {
    "ho chi minh": ["hcmc", "saigon", "ho chi minh", "district 1"],
    "hanoi":        ["hanoi", "ha noi", "hoan kiem"],
    "da nang":      ["da nang", "danang"],
    "hoi an":       ["hoi an", "hoian"],
    "nha trang":    ["nha trang", "nhatrang"],
    "phu quoc":     ["phu quoc", "phuquoc"],
    "ha long":      ["ha long", "halong"],
    "dalat":        ["da lat", "dalat"],
}

# Mock user database (in-memory, reset on restart)
USERS: dict = {
    "user_001": {"name": "Nguyen Van A", "balance_vnd": 5_000_000, "balance_usd": 200.0},
    "user_002": {"name": "Tran Thi B",   "balance_vnd": 3_200_000, "balance_usd": 128.0},
    "user_003": {"name": "Tourist_John", "balance_vnd": 10_000_000, "balance_usd": 400.0},
    "demo":     {"name": "Demo User",    "balance_vnd": 2_500_000, "balance_usd": 100.0},
}

TRANSACTIONS: dict = {uid: [] for uid in USERS}


# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────
class TransferRequest(BaseModel):
    from_user: str = Field(..., example="demo", description="ID người gửi")
    to_user: str   = Field(..., example="user_001", description="ID người nhận")
    amount_vnd: float = Field(..., gt=0, le=50_000_000, example=500_000,
                              description="Số tiền VND (tối đa 50,000,000)")
    note: Optional[str] = Field(None, example="Trả tiền ăn sáng", description="Ghi chú")


class TransferResponse(BaseModel):
    tx_id: str
    status: str
    from_user: str
    to_user: str
    amount_vnd: float
    amount_usd: Optional[float]
    note: Optional[str]
    timestamp: str
    message: str


class BalanceResponse(BaseModel):
    user_id: str
    name: str
    balance_vnd: float
    balance_usd: float
    last_updated: str


class ExchangeRate(BaseModel):
    currency: str
    buy_cash: Optional[float]
    buy_transfer: Optional[float]
    sell: Optional[float]


class ExchangeRatesResponse(BaseModel):
    source: str
    timestamp: str
    rates: List[ExchangeRate]
    usd_to_vnd: Optional[float]


class ScamRiskResponse(BaseModel):
    overall_risk_score: float = Field(..., description="0–100, cao = nguy hiểm hơn")
    risk_level: str           = Field(..., description="Low / Medium / High / Critical")
    negative_review_pct: float
    top_scam_types: List[dict]
    total_reviews_analyzed: int
    data_source: str
    recommendation: str


class LocationRiskResponse(BaseModel):
    location: str
    risk_score: float
    risk_level: str
    review_count: int
    common_scams: List[str]
    recent_alerts: List[str]


class ScamAlert(BaseModel):
    title: str
    scam_type: str
    sentiment_score: float
    risk_score: float
    date: str
    source_url: str


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def score_to_level(score: float) -> str:
    if score < 20:  return "Low"
    if score < 40:  return "Medium"
    if score < 60:  return "High"
    return "Critical"


async def fetch_vietcombank_rates() -> List[ExchangeRate]:
    """Fetch tỷ giá từ Vietcombank XML portal."""
    url = "https://portal.vietcombank.com.vn/Usercontrols/TVPortal.TyGia/pXML.aspx"
    proxy = f"https://api.allorigins.win/raw?url={url}"
    rates = []
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(proxy)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for ex in root.findall(".//Exrate"):
                def safe_float(val):
                    try:
                        return float(str(val).replace(",", "")) if val else None
                    except Exception:
                        return None
                rates.append(ExchangeRate(
                    currency=ex.get("CurrencyCode", ""),
                    buy_cash=safe_float(ex.get("Buy")),
                    buy_transfer=safe_float(ex.get("Transfer")),
                    sell=safe_float(ex.get("Sell")),
                ))
    except Exception as e:
        # Fallback rates nếu Vietcombank không trả lời
        rates = [
            ExchangeRate(currency="USD", buy_cash=24900, buy_transfer=24950, sell=25200),
            ExchangeRate(currency="EUR", buy_cash=27000, buy_transfer=27100, sell=27500),
            ExchangeRate(currency="GBP", buy_cash=31000, buy_transfer=31100, sell=31600),
            ExchangeRate(currency="JPY", buy_cash=161,   buy_transfer=162,   sell=168),
            ExchangeRate(currency="THB", buy_cash=660,   buy_transfer=665,   sell=700),
            ExchangeRate(currency="SGD", buy_cash=18500, buy_transfer=18600, sell=19000),
        ]
        print(f"⚠️  Vietcombank fetch failed: {e}. Using fallback rates.")
    return rates


def get_scam_stats() -> dict:
    """Tính thống kê scam từ DataFrame đã load."""
    if not SCAM_DATA_LOADED or df.empty:
        # Fallback stats
        return {
            "neg_pct": 35.5,
            "avg_risk": 12.7,
            "top_types": [
                {"type": "Currency / Change", "count": 53, "avg_sentiment": -0.28},
                {"type": "Overcharging",      "count": 44, "avg_sentiment": -0.22},
                {"type": "Accommodation",     "count": 33, "avg_sentiment": -0.19},
                {"type": "Theft / Pickpocket","count": 13, "avg_sentiment": -0.45},
                {"type": "Fake Grab / Taxi",  "count": 13, "avg_sentiment": -0.31},
            ],
            "total": 321,
        }

    neg_pct  = (df["sentiment"] == "Negative").mean() * 100
    avg_risk = df["risk_score"].mean()

    top_types = []
    for stype in SCAM_TYPES:
        if stype in df.columns:
            subset = df[df[stype] == 1]
            if len(subset) > 0:
                top_types.append({
                    "type": stype,
                    "count": int(subset[stype].sum()),
                    "avg_sentiment": round(float(subset["ensemble_score"].mean()), 3),
                })
    top_types.sort(key=lambda x: x["count"], reverse=True)

    return {
        "neg_pct":  round(neg_pct, 1),
        "avg_risk": round(avg_risk, 1),
        "top_types": top_types[:5],
        "total": len(df),
    }


def get_location_risk(location: str) -> dict:
    """Risk score cho 1 địa điểm cụ thể."""
    location_lower = location.lower()
    matched_key = None
    for key, keywords in LOCATION_KEYWORDS.items():
        if any(kw in location_lower for kw in keywords) or location_lower in keywords:
            matched_key = key
            break

    if not SCAM_DATA_LOADED or df.empty or matched_key is None:
        # Generic fallback
        return {
            "review_count": 0,
            "risk_score": 20.0,
            "common_scams": ["Overcharging", "Fake Grab / Taxi"],
            "recent_titles": [],
        }

    # Filter reviews matching location
    mask = df["title"].str.lower().str.contains(
        "|".join(LOCATION_KEYWORDS[matched_key]), na=False
    )
    subset = df[mask]

    if len(subset) == 0:
        return {
            "review_count": 0,
            "risk_score": get_scam_stats()["avg_risk"],
            "common_scams": ["Overcharging"],
            "recent_titles": [],
        }

    common_scams = []
    for stype in SCAM_TYPES:
        if stype in subset.columns and subset[stype].sum() > 0:
            common_scams.append(stype)

    return {
        "review_count": len(subset),
        "risk_score": round(float(subset["risk_score"].mean()), 1),
        "common_scams": common_scams[:3],
        "recent_titles": subset.sort_values("date", ascending=False)["title"].head(3).tolist(),
    }


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Health check")
async def root():
    return {
        "service": "TravelPay Vietnam API",
        "version": "1.0.0",
        "status": "running",
        "scam_data_loaded": SCAM_DATA_LOADED,
        "total_reviews": len(df) if SCAM_DATA_LOADED else 0,
        "docs": "/docs",
    }


@app.get(
    "/exchange-rates",
    response_model=ExchangeRatesResponse,
    tags=["Currency"],
    summary="Tỷ giá ngoại tệ realtime từ Vietcombank",
    description="""
Fetch tỷ giá mua/bán từ Vietcombank XML API.  
Thay thế Stripe/Plaid không hỗ trợ VND.  
Trả về mua tiền mặt, mua chuyển khoản, và bán cho các đồng tiền chính.
    """,
)
async def get_exchange_rates():
    rates = await fetch_vietcombank_rates()
    usd_rate = next((r.sell for r in rates if r.currency == "USD"), None)
    return ExchangeRatesResponse(
        source="Vietcombank (portal.vietcombank.com.vn)",
        timestamp=datetime.utcnow().isoformat() + "Z",
        rates=rates,
        usd_to_vnd=usd_rate,
    )


@app.post(
    "/transfer",
    response_model=TransferResponse,
    tags=["Transactions"],
    summary="Chuyển tiền VND giữa tài khoản",
    description="""
Thực hiện giao dịch chuyển tiền VND.  
**Lưu ý demo**: dữ liệu in-memory, reset khi restart server.  

Tự động convert sang USD dựa trên tỷ giá Vietcombank realtime.
    """,
)
async def transfer(req: TransferRequest):
    # Validate users
    if req.from_user not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{req.from_user}' không tồn tại")
    if req.to_user not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{req.to_user}' không tồn tại")
    if req.from_user == req.to_user:
        raise HTTPException(status_code=400, detail="Không thể chuyển tiền cho chính mình")

    sender = USERS[req.from_user]
    receiver = USERS[req.to_user]

    if sender["balance_vnd"] < req.amount_vnd:
        raise HTTPException(
            status_code=400,
            detail=f"Số dư không đủ. Hiện có: {sender['balance_vnd']:,.0f} VND"
        )

    # Fetch tỷ giá để convert
    rates = await fetch_vietcombank_rates()
    usd_sell = next((r.sell for r in rates if r.currency == "USD"), 25200)
    amount_usd = round(req.amount_vnd / usd_sell, 2)

    # Execute
    sender["balance_vnd"]   -= req.amount_vnd
    sender["balance_usd"]   -= amount_usd
    receiver["balance_vnd"] += req.amount_vnd
    receiver["balance_usd"] += amount_usd

    tx = {
        "tx_id": f"TX-{uuid.uuid4().hex[:8].upper()}",
        "from_user": req.from_user,
        "to_user": req.to_user,
        "amount_vnd": req.amount_vnd,
        "amount_usd": amount_usd,
        "note": req.note,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "completed",
    }
    TRANSACTIONS[req.from_user].append(tx)
    TRANSACTIONS[req.to_user].append({**tx, "amount_vnd": req.amount_vnd, "direction": "received"})

    return TransferResponse(
        **tx,
        message=f"✅ Chuyển {req.amount_vnd:,.0f} VND (~${amount_usd}) thành công"
    )


@app.get(
    "/balance/{user_id}",
    response_model=BalanceResponse,
    tags=["Transactions"],
    summary="Số dư tài khoản theo VND",
)
async def get_balance(user_id: str):
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' không tồn tại")
    user = USERS[user_id]
    return BalanceResponse(
        user_id=user_id,
        name=user["name"],
        balance_vnd=user["balance_vnd"],
        balance_usd=user["balance_usd"],
        last_updated=datetime.utcnow().isoformat() + "Z",
    )


@app.get(
    "/transactions/{user_id}",
    tags=["Transactions"],
    summary="Lịch sử giao dịch",
)
async def get_transactions(
    user_id: str,
    limit: int = Query(10, ge=1, le=50, description="Số giao dịch trả về"),
):
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' không tồn tại")
    txs = TRANSACTIONS.get(user_id, [])
    return {
        "user_id": user_id,
        "total": len(txs),
        "transactions": txs[-limit:][::-1],  # newest first
    }


@app.get(
    "/scam-risk",
    response_model=ScamRiskResponse,
    tags=["AI Scam Intelligence"],
    summary="AI Scam Risk Score toàn quốc",
    description="""
**Alternative Data Analysis (Part A)**  

Risk score được tính từ **321 Reddit reviews** (r/vietnam, r/travel, r/solotravel)  
qua **Ensemble Sentiment Model**: VADER (70%) + TextBlob (30%) + custom scam lexicon.

Trả về:
- Overall risk score (0–100)
- Top 5 loại scam phổ biến nhất  
- Phần trăm review tiêu cực
- Recommendation cho user
    """,
)
async def get_scam_risk():
    stats = get_scam_stats()
    neg_pct   = stats["neg_pct"]
    avg_risk  = stats["avg_risk"]
    top_types = stats["top_types"]

    # Composite risk score
    risk_score = round(neg_pct * 0.6 + avg_risk * 0.4, 1)
    risk_level = score_to_level(risk_score)

    if risk_level == "Low":
        rec = "Khu vực tương đối an toàn. Vẫn nên dùng Grab thay taxi ở đường phố."
    elif risk_level == "Medium":
        rec = "Cẩn thận với tiền lẻ khi đổi tiền và luôn confirm giá trước khi đi xe."
    elif risk_level == "High":
        rec = "Nhiều báo cáo scam. Chỉ dùng app chính thức, tránh đổi tiền vỉa hè."
    else:
        rec = "Nguy hiểm cao. Chỉ giao dịch qua TravelPay, không mang tiền mặt lớn."

    return ScamRiskResponse(
        overall_risk_score=risk_score,
        risk_level=risk_level,
        negative_review_pct=neg_pct,
        top_scam_types=top_types,
        total_reviews_analyzed=stats["total"],
        data_source="Reddit (r/vietnam, r/travel, r/solotravel) — Ensemble NLP Model",
        recommendation=rec,
    )


@app.get(
    "/scam-risk/location",
    response_model=LocationRiskResponse,
    tags=["AI Scam Intelligence"],
    summary="Scam risk theo địa điểm cụ thể",
    description="""
Risk score cho một thành phố/khu vực tại Việt Nam.  
Hỗ trợ: `ho chi minh`, `hanoi`, `da nang`, `hoi an`, `nha trang`, `phu quoc`, `ha long`, `dalat`
    """,
)
async def get_location_risk(
    location: str = Query(..., example="hoi an", description="Tên địa điểm"),
):
    data = get_location_risk(location)
    stats = get_scam_stats()

    # If no specific data, use average
    risk_score = data["risk_score"] if data["review_count"] > 0 else stats["avg_risk"]

    return LocationRiskResponse(
        location=location.title(),
        risk_score=risk_score,
        risk_level=score_to_level(risk_score),
        review_count=data["review_count"],
        common_scams=data["common_scams"] if data["common_scams"] else ["Overcharging"],
        recent_alerts=data["recent_titles"],
    )


@app.get(
    "/scam-alerts",
    tags=["AI Scam Intelligence"],
    summary="Top cảnh báo scam mới nhất",
    description="Trả về N bài viết có risk score cao nhất từ data thực tế.",
)
async def get_scam_alerts(
    limit: int = Query(5, ge=1, le=20),
    scam_type: Optional[str] = Query(None, example="Fake Grab / Taxi",
                                      description="Lọc theo loại scam"),
):
    if not SCAM_DATA_LOADED or df.empty:
        return {"alerts": [], "message": "Scam data not available"}

    subset = df[df["sentiment"] == "Negative"].copy()

    if scam_type and scam_type in SCAM_TYPES and scam_type in subset.columns:
        subset = subset[subset[scam_type] == 1]

    top = subset.nlargest(limit, "risk_score")

    alerts = []
    for _, row in top.iterrows():
        # Find dominant scam type
        dominant = "General"
        max_val = 0
        for stype in SCAM_TYPES:
            if stype in row and row[stype] > max_val:
                max_val = row[stype]
                dominant = stype

        alerts.append(ScamAlert(
            title=str(row["title"])[:100],
            scam_type=dominant,
            sentiment_score=round(float(row["ensemble_score"]), 3),
            risk_score=round(float(row["risk_score"]), 1),
            date=str(row["date"])[:10] if pd.notna(row["date"]) else "unknown",
            source_url=str(row.get("url", "")),
        ))

    return {
        "total_alerts": len(alerts),
        "filter_applied": scam_type,
        "alerts": [a.dict() for a in alerts],
    }
