# TravelPay Vietnam — Custom OpenAPI Backend

Backend server cho app chuyển tiền du lịch. Tích hợp tỷ giá Vietcombank + AI Scam Intelligence.

---

## Cấu trúc file

```
├── main.py                              ← FastAPI server (file chính)
├── vietnam_scam_sentiment_results.csv  ← AI data từ Part A (đặt cùng thư mục)
├── requirements.txt
├── Procfile                             ← cho Railway/Render
└── README.md
```

---

## Chạy local

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Đảm bảo file CSV trong cùng thư mục với main.py
# vietnam_scam_sentiment_results.csv

# 3. Chạy server
uvicorn main:app --reload --port 8000
```

Mở trình duyệt: **http://localhost:8000/docs** → Swagger UI

---

## Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| GET | `/` | Health check |
| GET | `/exchange-rates` | Tỷ giá Vietcombank realtime |
| POST | `/transfer` | Chuyển tiền VND |
| GET | `/balance/{user_id}` | Số dư tài khoản |
| GET | `/transactions/{user_id}` | Lịch sử giao dịch |
| GET | `/scam-risk` | AI scam risk toàn quốc |
| GET | `/scam-risk/location?location=hoi+an` | Risk theo địa điểm |
| GET | `/scam-alerts?limit=5` | Top cảnh báo mới nhất |

**Demo users**: `demo`, `user_001`, `user_002`, `user_003`

---

## Deploy lên Railway (miễn phí)

```bash
# 1. Push code lên GitHub repo
git init
git add .
git commit -m "Initial TravelPay API"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main

# 2. Vào https://railway.app
# 3. New Project → Deploy from GitHub repo → chọn repo
# 4. Railway tự detect Procfile, deploy trong ~2 phút
# 5. Settings → Generate Domain → lấy URL public
```

URL sau khi deploy: `https://<your-app>.railway.app/docs`

---

## Tích hợp Google AI Studio (Function Calling)

Vào **https://aistudio.google.com** → chọn model Gemini → tab **"Tools"**

### Khai báo 3 functions:

#### 1. check_scam_risk
```json
{
  "name": "check_scam_risk",
  "description": "Lấy AI scam risk score cho một địa điểm tại Việt Nam. Gọi khi user hỏi về độ an toàn, scam, hay cảnh báo ở một khu vực.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "Tên thành phố hoặc địa điểm tại Việt Nam. Ví dụ: hoi an, da nang, ho chi minh"
      }
    },
    "required": ["location"]
  }
}
```

#### 2. transfer_money
```json
{
  "name": "transfer_money",
  "description": "Chuyển tiền VND giữa hai tài khoản. Gọi khi user nói 'chuyển tiền', 'gửi tiền', 'thanh toán'.",
  "parameters": {
    "type": "object",
    "properties": {
      "from_user": {
        "type": "string",
        "description": "ID tài khoản người gửi. Mặc định dùng 'demo'"
      },
      "to_user": {
        "type": "string",
        "description": "ID tài khoản người nhận"
      },
      "amount_vnd": {
        "type": "number",
        "description": "Số tiền VND cần chuyển. Ví dụ 500000 = 500k VND"
      },
      "note": {
        "type": "string",
        "description": "Ghi chú giao dịch (tùy chọn)"
      }
    },
    "required": ["from_user", "to_user", "amount_vnd"]
  }
}
```

#### 3. get_exchange_rate
```json
{
  "name": "get_exchange_rate",
  "description": "Lấy tỷ giá VND từ Vietcombank. Gọi khi user hỏi tỷ giá, đổi tiền, hay muốn biết giá trị USD/EUR sang VND.",
  "parameters": {
    "type": "object",
    "properties": {
      "currency": {
        "type": "string",
        "description": "Mã ngoại tệ: USD, EUR, GBP, JPY, THB, SGD"
      }
    },
    "required": ["currency"]
  }
}
```

### System prompt gợi ý cho AI Studio:

```
Bạn là trợ lý tài chính của TravelPay Vietnam — app chuyển tiền cho khách du lịch.

Khi user hỏi về:
- Chuyển tiền / gửi tiền → gọi transfer_money()
- Tỷ giá, đổi tiền → gọi get_exchange_rate()
- Scam, an toàn, cảnh báo → gọi check_scam_risk()

Sau khi nhận kết quả API:
- Nếu risk_level = "High" hoặc "Critical" → cảnh báo rõ ràng và gợi ý dùng TravelPay thay tiền mặt
- Luôn hiển thị số tiền VND với format có dấu phẩy (500,000 VND)
- Trả lời bằng tiếng Việt, thân thiện và ngắn gọn
```

### Cách gọi API từ AI Studio:

Khi functions đã khai báo, trong System Instructions thêm base URL:
```
API Base URL: https://<your-app>.railway.app
Khi cần gọi check_scam_risk(location="hoi an"), 
thực hiện GET https://<your-app>.railway.app/scam-risk/location?location=hoi+an
```

---

## Ví dụ demo cho hội đồng

**User gõ:** "Khu vực Hội An có an toàn không? Tôi muốn chuyển 500k cho bạn tôi"

**Flow:**
1. Gemini nhận → parse intent → 2 function calls song song
2. `check_scam_risk("hoi an")` → `GET /scam-risk/location?location=hoi+an`
3. `transfer_money("demo", "user_001", 500000)` → `POST /transfer`
4. Gemini tổng hợp → "Hội An có 3 reviews liên quan, risk Medium. Đã chuyển 500,000 VND thành công! ⚠️ Lưu ý: khu vực này hay gặp scam Fake Grab, hãy đặt xe qua app."
