import os,sys,joblib
from pprint import pprint
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from train_save import train_and_save_model
import gradio as gr

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ==========================================
# 1. 載入模型與狀態管理
# ==========================================

model_path:str = os.path.join(current_dir, "salary_model.joblib")
MODEL_STATE:dict = {}

def load_model_state():
    global MODEL_STATE
    if not os.path.exists(model_path):
        print("未檢測到模型檔案，正在自動執行訓練以生成 salary_model.joblib...")
        try:
            train_and_save_model()
        except Exception as e:
            raise RuntimeError(f"自動訓練模型失敗: {str(e)}")

    # 載入模型與相關元數據
    model_data:dict = joblib.load(model_path)
    #pprint(model_data)
    MODEL_STATE.clear()
    MODEL_STATE.update({
            "model": model_data["model"],
            "oe": model_data.get("oe"),
            "ohe": model_data["ohe"],
            "scaler": model_data["scaler"],
            "r2": model_data.get("r2", 0.8463),
            "coef": model_data.get("coef",[]),
            "intercept": model_data.get("intercept", 51.2286),
            "feature_names": model_data.get("feature_names",['YearsExperience', 'EducationLevel', 'City_城市A', 'City_城市B', 'City_城市C']),
            "feature_coefs": model_data.get("feature_coefs", {}),
            "model_type": model_data.get("model_type","LinearRegression"),
            "alpha": model_data.get("alpha", 1.0),
            "train_time": model_data.get("train_time", 0.01),
            "test_size": model_data.get("test_size", 0.2),
            "random_state": model_data.get("random_state", 76)
        }
    )

    print("模型與預處理器成功載入！目前 R² Score：", MODEL_STATE["r2"])

# =========================================
# 2.建立 FastAPI 應用與 Pydantic 格式定義
# =========================================

api_app = FastAPI(
    title = "薪資預測多元線性迴歸 API",
    description="這是一個結合 FastAPI 與 Gradio 的機器學習部署服務。提供薪資預測端點與線上模型訓練端點。",
    version="1.0.0"
)

# --- Pydantic 預測模型 ---

class SalaryInput(BaseModel):
    years_experience: float = Field(..., description="工作年資 (年，通常為 1.0 ~ 10.0)", ge=0.0, le=50.0)
    education_level: str = Field(..., description="學歷 (大學、碩士以上、高中以下)")
    city: str = Field(..., description="工作城市 (城市A、城市B、城市C)")

    model_config ={
        "json_schema_extra":{
            "example":{
                "years_experience": 5.3,
                "education_level": "碩士以上",
                "city": "城市A"
            }
        }
    }

class SalaryOutput(BaseModel):
    predicted_salary: float = Field(..., description="預測月薪 (k / 千元)")
    estimated_annual_salary: float = Field(..., description="估計年薪 (k / 千元，以 14 個月估算)")

class TrainConfig(BaseModel):
    test_size: float = Field(0.2, description="測試集分割比例", ge=0.1, le=0.5)
    random_state: int = Field(76, description="隨機種子", ge=0)
    model_type:str = Field("LinearRegression", description="模型演算法類型 (LinearRegression, Lasso, Ridge)")
    alpha:float = Field(1.0, description="正則化強度 alpha (適用於 Lasso 與 Ridge)", ge=0.001, le=100.0)

class TrainResult(BaseModel):
    status: str = Field(..., description="執行結果狀態")
    r2: float = Field(..., description="測試集 R-squared 決定係數")
    coef: list[float] = Field(..., description="特徵權重係數列表")
    intercept: float = Field(..., description="截距")
    feature_coefs:dict[str, float] = Field(..., description="特徵及其權重映射")
    model_type: str = Field(..., description="模型演算法類型")
    alpha: float = Field(..., description="正則化強度 alpha")
    train_time: float = Field(..., description="訓練耗時 (秒)")
    message: str = Field(..., description="提示訊息")

@api_app.post("/predict",response_model=SalaryOutput)
def predict_api(payload:SalaryInput):
    """
    預測端點：接收年資、學歷、城市，進行編碼與標準化後，回傳模型預測的月薪與估計年薪。
    """
    try:
        ohe = MODEL_STATE["ohe"]
        scaler = MODEL_STATE["scaler"]
        model = MODEL_STATE["model"]
        oe:OrdinalEncoder = MODEL_STATE.get("oe") # type: ignore

        if oe is not None:
            try:
                edu_encoded = int(oe.transform(pd.DataFrame([[payload.education_level]], columns=["EducationLevel"]))[0][0])
            except ValueError:
                valid_cats = list(oe.categories_[0]) # type: ignore
                raise HTTPException(
                    status_code = 400,
                    detail= f"未知的學歷:{payload.education_level}. 可接受的值為:{valid_cats}"
                    )

        if ohe is not None:
            try:
                city_encoded = ohe.transform(pd.DataFrame([[payload.city]],columns=["City"]))[0]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"未知的城市: {payload.city}。可接受的值為: 城市A, 城市B, 城市C"
                )

        feature_names = MODEL_STATE["feature_names"]
        features_df = pd.DataFrame([[payload.years_experience, edu_encoded] + list(city_encoded)], columns=feature_names)

        features_scaled = scaler.transform(features_df)

        pred_val = float(model.predict(features_scaled)[0])

        return SalaryOutput(
            predicted_salary=pred_val,
            estimated_annual_salary=pred_val * 14
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗:{str(e)}")

@api_app.post("/train", response_model=TrainResult)
def train_endpoint(config:TrainConfig):
    """
    訓練端點：傳入測試集比例、隨機種子、模型類型與 alpha，線上重新訓練模型，並即時更新服務所使用的模型。
    """
    try:
        res = train_and_save_model(
            test_size = config.test_size,
            random_state = config.random_state,
            model_type = config.model_type,
            alpha = config.alpha
        )
        load_model_state()

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"線上訓練失敗:{str(e)}")
   
# --- 建立 Gradio UI Blocks 布局 ---
with gr.Blocks(
    title="💼 薪資預測多元線性迴歸平台"
) as demo:
    gr.Markdown(
        """
        # 💼 薪資預測多元線性迴歸教學與部署平台
        本系統展示了機器學習模型部署的**完整生命週期**。此服務底層使用 **FastAPI** 驅動，提供標準化 RESTful API，並結合 **Gradio** 開發了互動式 Web 介面。
        * 🔮 **即時預測分頁**：輸入您的工作年資、學歷與工作城市，即時透過多元線性迴歸模型取得月薪與年薪估計。
        * ⚙️ **線上訓練與公式分頁**：可線上調整測試集切分比例與隨機種子，即時訓練模型，並動態展示擬合後的**數學迴歸方程式**與特徵權重係數。
        """
    )
# ==========================================
# 4. 融合 Gradio 與自訂 API 路由
# ==========================================

# 1. 產生 Gradio 的 FastAPI 應用實例
#    (Gradio 預設停用 docs_url，透過 app_kwargs 重新啟用 Swagger UI)
app = gr.routes.App.create_app(demo, app_kwargs={"docs_url": "/docs"})

# 2. 合併 API 路由：將 api_app 中的所有自訂 API 路由 (/predict, /train) 併入
app.include_router(api_app.router)


if __name__ == "__main__":
    import uvicorn
    # Render 會透過 PORT 環境變數指定對外埠號；本地開發預設 8000
    port = int(os.environ.get("PORT", 8000))
    # 本地開發可設定環境變數 RELOAD=true 啟用熱重載；Render 生產環境維持關閉
    reload = os.environ.get("RELOAD", "").lower() == "true"
    print(f"使用 uvicorn 啟動伺服器 (port={port}, reload={reload})...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)





