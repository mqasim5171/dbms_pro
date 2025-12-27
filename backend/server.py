from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import jwt
import os
import logging
import uuid
import random
from pathlib import Path
from fastapi import UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# ✅ BBN imports
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------- ENV & DB SETUP ----------------

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# ---------------- SECURITY / AUTH SETUP ----------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------- FASTAPI APP & CORS ----------------

app = FastAPI(title="Real Estate Listing & Booking API")
api_router = APIRouter(prefix="/api")



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


allowed_origins = [
    "http://localhost:3000",
    "https://fancy-gelato-c39f14.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "Real Estate Listing & Booking API"}


# ---------------- ROLES & DEPENDENCIES ----------------

class UserRole(str, Enum):
    buyer = "buyer"
    owner = "owner"
    admin = "admin"
    user = "user"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ---------------- Pydantic MODELS ----------------

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None
    role: UserRole = UserRole.buyer


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class User(BaseModel):
    id: str
    email: EmailStr
    name: str
    phone: Optional[str] = None
    role: UserRole
    created_at: str


class CityCreate(BaseModel):
    name: str = Field(..., alias="city_name")


class City(BaseModel):
    id: str
    city_name: str


class PropertyTypeCreate(BaseModel):
    type_name: str


class PropertyType(BaseModel):
    id: str
    type_name: str


class PropertyCreate(BaseModel):
    title: str
    description: str
    price: float
    city_id: Optional[str] = None
    property_type_id: Optional[str] = None
    location: str
    address: str
    bedrooms: int
    bathrooms: int
    area: float
    status: str = "available"
    image_url: str
    amenities: List[str] = []


class Property(BaseModel):
    id: str
    title: str
    description: str
    price: float
    city_id: Optional[str] = None
    property_type_id: Optional[str] = None
    location: str
    address: str
    bedrooms: int
    bathrooms: int
    area: float
    status: str
    image_url: str
    amenities: List[str]
    created_at: str
    owner_id: str


class BookingCreate(BaseModel):
    property_id: str
    check_in: str
    check_out: Optional[str] = None
    guests: int
    message: Optional[str] = None
    total_amount: Optional[float] = None


class Booking(BaseModel):
    id: str
    property_id: str
    user_id: str
    user_name: str
    user_email: str
    check_in: str
    check_out: Optional[str] = None
    guests: int
    message: Optional[str] = None
    status: str = "pending"
    total_amount: float
    created_at: str


class PaymentMethod(str, Enum):
    cash = "cash"
    bank_transfer = "bank_transfer"
    online = "online"


class PaymentCreate(BaseModel):
    booking_id: str
    amount: float
    method: PaymentMethod


class Payment(BaseModel):
    id: str
    booking_id: str
    amount: float
    method: PaymentMethod
    payment_date: str
    status: str
    created_at: str


class ReviewCreate(BaseModel):
    property_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class Review(BaseModel):
    id: str
    property_id: str
    user_id: str
    user_name: str
    rating: int
    comment: Optional[str] = None
    created_at: str


class FavoriteCreate(BaseModel):
    property_id: str


class Favorite(BaseModel):
    id: str
    user_id: str
    property_id: str
    created_at: str


# ---------------- SMALL HELPERS ----------------

def normalize_status(v: Optional[str]) -> str:
    return (v or "").strip().lower()


# ---------------- BBN PERSISTENCE HELPERS ----------------

async def save_bbn_status():
    await db.system.update_one(
        {"key": "bbn_recommender"},
        {"$set": {
            "trained_at": BBN_CACHE["trained_at"],
            "rows_used": BBN_CACHE["rows_used"]
        }},
        upsert=True
    )


async def load_bbn_status():
    status = await db.system.find_one({"key": "bbn_recommender"})
    if status:
        BBN_CACHE["trained_at"] = status.get("trained_at")
        BBN_CACHE["rows_used"] = status.get("rows_used", 0)


# ---------------- BBN RECOMMENDER (DWDM Integration) ----------------

BBN_FEATURES = [
    "pref_city", "pref_type", "pref_price_bucket",
    "prop_city", "prop_type", "price_bucket", "beds_bucket"
]
BBN_TARGET = "liked"

BBN_CACHE: Dict[str, Any] = {
    "infer": None,          # VariableElimination
    "model": None,          # DiscreteBayesianNetwork
    "trained_at": None,
    "rows_used": 0,
}

def _bucket_price(price: float) -> str:
    if price < 150000:
        return "low"
    if price < 500000:
        return "mid"
    return "high"

def _bucket_beds(b: int) -> str:
    if b <= 2:
        return "1-2"
    if b <= 4:
        return "3-4"
    return "5+"

def _most_common(values, default="unknown"):
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]

def _prob_one(q) -> float:
    states = list(q.state_names[BBN_TARGET])
    if "1" in states:
        return float(q.values[states.index("1")])
    return float(q.values[-1])

async def _build_user_prefs(user_id: str) -> dict:
    favs = await db.favorites.find({"user_id": user_id}, {"_id": 0}).to_list(5000)
    books = await db.bookings.find({"user_id": user_id}, {"_id": 0}).to_list(5000)
    liked_prop_ids = set([f["property_id"] for f in favs] + [b["property_id"] for b in books])

    if not liked_prop_ids:
        return {"pref_city": "unknown", "pref_type": "unknown", "pref_price_bucket": "unknown"}

    props = await db.properties.find({"id": {"$in": list(liked_prop_ids)}}, {"_id": 0}).to_list(5000)

    cities = [str(p.get("city_id", "unknown")) for p in props]
    types = [str(p.get("property_type_id", "unknown")) for p in props]
    prices = [_bucket_price(float(p.get("price", 0))) for p in props]

    return {
        "pref_city": _most_common(cities),
        "pref_type": _most_common(types),
        "pref_price_bucket": _most_common(prices),
    }

async def _train_bbn_from_db(max_pos_per_user: int = 10, neg_ratio: int = 3) -> dict:
    """
    Builds hard-negative dataset from Mongo and trains a structured BBN.
    """
    users = await db.users.find({}, {"_id": 0, "id": 1, "role": 1}).to_list(10000)
    props = await db.properties.find({}, {"_id": 0, "id": 1, "city_id": 1, "property_type_id": 1, "price": 1, "bedrooms": 1}).to_list(20000)
    favs = await db.favorites.find({}, {"_id": 0, "user_id": 1, "property_id": 1}).to_list(200000)
    books = await db.bookings.find({}, {"_id": 0, "user_id": 1, "property_id": 1}).to_list(200000)

    liked_pairs = set()
    for x in favs:
        liked_pairs.add((x["user_id"], x["property_id"]))
    for x in books:
        liked_pairs.add((x["user_id"], x["property_id"]))

    if not liked_pairs:
        raise HTTPException(status_code=400, detail="No favorites/bookings found. Run /api/admin/seed-reco first.")

    prop_map = {p["id"]: p for p in props}

    hist_city = defaultdict(list)
    hist_type = defaultdict(list)
    hist_price = defaultdict(list)

    for (uid, pid) in liked_pairs:
        p = prop_map.get(pid)
        if not p:
            continue
        hist_city[uid].append(str(p.get("city_id", "unknown")))
        hist_type[uid].append(str(p.get("property_type_id", "unknown")))
        hist_price[uid].append(_bucket_price(float(p.get("price", 0))))

    buyers = [u for u in users if str(u.get("role", "")).lower() == "buyer"]
    if not buyers:
        raise HTTPException(status_code=400, detail="No buyer users found. Run /api/admin/seed-reco first.")

    random.seed(42)
    rows = []

    for u in buyers:
        uid = u["id"]
        pref_city = _most_common(hist_city[uid])
        pref_type = _most_common(hist_type[uid])
        pref_price = _most_common(hist_price[uid])

        user_pos = [pid for (xuid, pid) in liked_pairs if xuid == uid and pid in prop_map]
        random.shuffle(user_pos)
        user_pos = user_pos[:max_pos_per_user]
        if not user_pos:
            continue

        for pid in user_pos:
            p = prop_map[pid]

            # positive
            rows.append({
                "pref_city": pref_city,
                "pref_type": pref_type,
                "pref_price_bucket": pref_price,
                "prop_city": str(p.get("city_id", "unknown")),
                "prop_type": str(p.get("property_type_id", "unknown")),
                "price_bucket": _bucket_price(float(p.get("price", 0))),
                "beds_bucket": _bucket_beds(int(p.get("bedrooms", 0))),
                "liked": "1",
            })

            # hard negatives (mismatch)
            hard_candidates = []
            for q in props:
                if q["id"] == pid:
                    continue
                q_city = str(q.get("city_id", "unknown"))
                q_type = str(q.get("property_type_id", "unknown"))
                q_price = _bucket_price(float(q.get("price", 0)))

                mismatch = (
                    (pref_city != "unknown" and q_city != pref_city) or
                    (pref_type != "unknown" and q_type != pref_type) or
                    (pref_price != "unknown" and q_price != pref_price)
                )
                if mismatch:
                    hard_candidates.append(q)

            if hard_candidates:
                negs = random.sample(hard_candidates, k=min(len(hard_candidates), neg_ratio))
                for q in negs:
                    rows.append({
                        "pref_city": pref_city,
                        "pref_type": pref_type,
                        "pref_price_bucket": pref_price,
                        "prop_city": str(q.get("city_id", "unknown")),
                        "prop_type": str(q.get("property_type_id", "unknown")),
                        "price_bucket": _bucket_price(float(q.get("price", 0))),
                        "beds_bucket": _bucket_beds(int(q.get("bedrooms", 0))),
                        "liked": "0",
                    })

    if len(rows) < 200:
        raise HTTPException(status_code=400, detail="Not enough rows to train recommender. Increase interactions using seed-reco.")

    df = pd.DataFrame(rows).astype(str)

    # ✅ structured causal network
    edges = [
        ("pref_city", "prop_city"),
        ("pref_type", "prop_type"),
        ("pref_price_bucket", "price_bucket"),
        ("prop_city", BBN_TARGET),
        ("prop_type", BBN_TARGET),
        ("price_bucket", BBN_TARGET),
        ("beds_bucket", BBN_TARGET),
    ]

    model = DiscreteBayesianNetwork(edges)
    model.fit(df[BBN_FEATURES + [BBN_TARGET]], estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)

    return {"infer": infer, "model": model, "rows": int(len(df)), "edges": edges}

# ---------------- AUTH ROUTES ----------------

@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed_password = hash_password(user_data.password)

    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "password": hashed_password,
        "name": user_data.name,
        "phone": user_data.phone,
        "role": user_data.role.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(user_doc)

    access_token = create_access_token({"user_id": user_id, "email": user_data.email, "role": user_data.role.value})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "email": user_data.email, "name": user_data.name, "role": user_data.role.value},
    }

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"user_id": user["id"], "email": user["email"], "role": user["role"]})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "name": user["name"], "role": user["role"]},
    }

@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "email": current_user["email"], "name": current_user["name"], "role": current_user["role"]}

# ---------------- CITY & PROPERTY TYPE ROUTES ----------------

@api_router.get("/cities", response_model=List[City])
async def list_cities():
    return await db.cities.find({}, {"_id": 0}).to_list(1000)

@api_router.post("/cities", response_model=City)
async def create_city(city_data: CityCreate, current_user: dict = Depends(get_admin_user)):
    city_id = str(uuid.uuid4())
    doc = {"id": city_id, "city_name": city_data.city_name}
    await db.cities.insert_one(doc)
    return doc

@api_router.get("/property-types", response_model=List[PropertyType])
async def list_property_types():
    return await db.property_types.find({}, {"_id": 0}).to_list(1000)

@api_router.post("/property-types", response_model=PropertyType)
async def create_property_type(type_data: PropertyTypeCreate, current_user: dict = Depends(get_admin_user)):
    type_id = str(uuid.uuid4())
    doc = {"id": type_id, "type_name": type_data.type_name}
    await db.property_types.insert_one(doc)
    return doc

# ---------------- PROPERTY ROUTES ----------------

@api_router.get("/properties", response_model=List[Property])
async def get_properties(
    status: Optional[str] = None,
    city_id: Optional[str] = None,
    property_type_id: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
):
    query: dict = {}
    if status:
        query["status"] = status
    if city_id:
        query["city_id"] = city_id
    if property_type_id:
        query["property_type_id"] = property_type_id
    if bedrooms is not None:
        query["bedrooms"] = {"$gte": bedrooms}
    if bathrooms is not None:
        query["bathrooms"] = {"$gte": bathrooms}
    if min_price is not None or max_price is not None:
        pf = {}
        if min_price is not None:
            pf["$gte"] = min_price
        if max_price is not None:
            pf["$lte"] = max_price
        query["price"] = pf

    return await db.properties.find(query, {"_id": 0}).to_list(1000)

@api_router.get("/properties/{property_id}", response_model=Property)
async def get_property(property_id: str):
    doc = await db.properties.find_one({"id": property_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Property not found")
    return doc

@api_router.post("/properties", response_model=Property)
async def create_property(property_data: PropertyCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in [UserRole.owner.value, UserRole.admin.value]:
        raise HTTPException(status_code=403, detail="Only owners (dealers) or admin can create properties")

    property_id = str(uuid.uuid4())
    doc = {
        "id": property_id,
        **property_data.model_dump(),
        "owner_id": current_user["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.properties.insert_one(doc)
    return doc

@api_router.put("/properties/{property_id}", response_model=Property)
async def update_property(property_id: str, property_data: PropertyCreate, current_user: dict = Depends(get_current_user)):
    existing = await db.properties.find_one({"id": property_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Property not found")
    if existing["owner_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Not authorized to update this property")

    updated_doc = {**property_data.model_dump(), "owner_id": existing["owner_id"], "created_at": existing["created_at"]}
    await db.properties.update_one({"id": property_id}, {"$set": updated_doc})
    return await db.properties.find_one({"id": property_id}, {"_id": 0})

@api_router.delete("/properties/{property_id}")
async def delete_property(property_id: str, current_user: dict = Depends(get_current_user)):
    existing = await db.properties.find_one({"id": property_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Property not found")
    if existing["owner_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Not authorized to delete this property")

    await db.properties.delete_one({"id": property_id})
    return {"message": "Property deleted successfully"}

@api_router.get("/owner/my-properties", response_model=List[Property])
async def get_my_properties(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in [UserRole.owner.value, UserRole.admin.value]:
        raise HTTPException(status_code=403, detail="Only owners or admin can view this")

    query = {}
    if current_user["role"] == UserRole.owner.value:
        query["owner_id"] = current_user["id"]

    return await db.properties.find(query, {"_id": 0}).to_list(1000)

# ---------------- BOOKING ROUTES ----------------

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(current_user: dict = Depends(get_current_user)):
    if current_user["role"] == UserRole.admin.value:
        return await db.bookings.find({}, {"_id": 0}).to_list(1000)

    if current_user["role"] == UserRole.owner.value:
        owner_props = await db.properties.find({"owner_id": current_user["id"]}, {"id": 1}).to_list(1000)
        prop_ids = [p["id"] for p in owner_props]
        return await db.bookings.find({"property_id": {"$in": prop_ids}}, {"_id": 0}).to_list(1000)

    return await db.bookings.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)

@api_router.get("/bookings/{booking_id}", response_model=Booking)
async def get_booking(booking_id: str, current_user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if booking["user_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        prop = await db.properties.find_one({"id": booking["property_id"]})
        if not prop or prop["owner_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Not authorized to view this booking")

    return booking

@api_router.post("/bookings", response_model=Booking)
async def create_booking(booking_data: BookingCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in [UserRole.buyer.value, UserRole.admin.value]:
        raise HTTPException(status_code=403, detail="Only buyers or admin can create bookings")

    prop = await db.properties.find_one({"id": booking_data.property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    if normalize_status(prop.get("status")) != "available":
        raise HTTPException(status_code=400, detail="Property is not available for booking")

    if prop["owner_id"] == current_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot book your own property")

    booking_id = str(uuid.uuid4())
    total_amount = booking_data.total_amount or float(prop["price"])

    doc = {
        "id": booking_id,
        "property_id": booking_data.property_id,
        "check_in": booking_data.check_in,
        "check_out": booking_data.check_out,
        "guests": booking_data.guests,
        "message": booking_data.message,
        "user_id": current_user["id"],
        "user_name": current_user["name"],
        "user_email": current_user["email"],
        "status": "pending",
        "total_amount": total_amount,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.bookings.insert_one(doc)
    return doc

@api_router.put("/bookings/{booking_id}/status", response_model=Booking)
async def update_booking_status(booking_id: str, status: str, current_user: dict = Depends(get_current_user)):
    if status not in ["pending", "confirmed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    prop = await db.properties.find_one({"id": booking["property_id"]})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    if current_user["role"] != UserRole.admin.value and prop["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Only property owner or admin can update booking status")

    if booking.get("status") == "confirmed":
        raise HTTPException(status_code=400, detail="Booking already confirmed")

    if status == "confirmed":
        if normalize_status(prop.get("status")) != "available":
            raise HTTPException(status_code=400, detail="Property is no longer available")

        await db.bookings.update_one({"id": booking_id}, {"$set": {"status": "confirmed"}})
        await db.properties.update_one({"id": prop["id"]}, {"$set": {"status": "booked"}})
        await db.bookings.update_many(
            {"property_id": prop["id"], "id": {"$ne": booking_id}, "status": "pending"},
            {"$set": {"status": "cancelled"}},
        )
    else:
        await db.bookings.update_one({"id": booking_id}, {"$set": {"status": status}})

    return await db.bookings.find_one({"id": booking_id}, {"_id": 0})

@api_router.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, current_user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    if booking["user_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Not authorized to delete this booking")

    await db.bookings.delete_one({"id": booking_id})
    return {"message": "Booking deleted successfully"}

# ---------------- PAYMENT ROUTES ----------------

@api_router.get("/payments", response_model=List[Payment])
async def list_payments(current_user: dict = Depends(get_current_user)):
    if current_user["role"] == UserRole.admin.value:
        return await db.payments.find({}, {"_id": 0}).to_list(1000)

    bookings = await db.bookings.find({"user_id": current_user["id"]}, {"id": 1}).to_list(1000)
    booking_ids = [b["id"] for b in bookings]
    return await db.payments.find({"booking_id": {"$in": booking_ids}}, {"_id": 0}).to_list(1000)

@api_router.post("/payments", response_model=Payment)
async def create_payment(payment_data: PaymentCreate, current_user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": payment_data.booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if booking["user_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Not authorized to create payment for this booking")

    pay_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": pay_id,
        "booking_id": payment_data.booking_id,
        "amount": payment_data.amount,
        "method": payment_data.method.value,
        "payment_date": now_iso,
        "status": "paid",
        "created_at": now_iso,
    }
    await db.payments.insert_one(doc)

    await db.bookings.update_one({"id": payment_data.booking_id}, {"$set": {"status": "confirmed"}})

    prop = await db.properties.find_one({"id": booking["property_id"]})
    if prop:
        await db.properties.update_one({"id": prop["id"]}, {"$set": {"status": "booked"}})
        await db.bookings.update_many(
            {"property_id": prop["id"], "id": {"$ne": booking["id"]}, "status": "pending"},
            {"$set": {"status": "cancelled"}},
        )
    return doc

# ---------------- REVIEW ROUTES ----------------

@api_router.get("/properties/{property_id}/reviews", response_model=List[Review])
async def list_reviews_for_property(property_id: str):
    return await db.reviews.find({"property_id": property_id}, {"_id": 0}).to_list(1000)

@api_router.post("/reviews", response_model=Review)
async def create_review(review_data: ReviewCreate, current_user: dict = Depends(get_current_user)):
    review_id = str(uuid.uuid4())
    doc = {
        "id": review_id,
        "property_id": review_data.property_id,
        "user_id": current_user["id"],
        "user_name": current_user["name"],
        "rating": review_data.rating,
        "comment": review_data.comment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.reviews.insert_one(doc)
    return doc

# ---------------- FAVORITES (WISHLIST) ROUTES ----------------

@api_router.get("/favorites", response_model=List[Favorite])
async def list_favorites(current_user: dict = Depends(get_current_user)):
    return await db.favorites.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)

@api_router.post("/favorites", response_model=Favorite)
async def add_favorite(fav_data: FavoriteCreate, current_user: dict = Depends(get_current_user)):
    existing = await db.favorites.find_one({"user_id": current_user["id"], "property_id": fav_data.property_id})
    if existing:
        raise HTTPException(status_code=400, detail="Property already in favorites")

    fav_id = str(uuid.uuid4())
    doc = {
        "id": fav_id,
        "user_id": current_user["id"],
        "property_id": fav_data.property_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.favorites.insert_one(doc)
    return doc

@api_router.delete("/favorites/{favorite_id}")
async def remove_favorite(favorite_id: str, current_user: dict = Depends(get_current_user)):
    fav = await db.favorites.find_one({"id": favorite_id})
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite not found")
    if fav["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Not authorized to remove this favorite")

    await db.favorites.delete_one({"id": favorite_id})
    return {"message": "Favorite removed successfully"}

# ---------------- RECOMMENDATIONS ROUTES ----------------

@api_router.get("/recommendations")
async def get_recommendations(limit: int = 10, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != UserRole.buyer.value:
        raise HTTPException(status_code=403, detail="Recommendations are for buyers only")

    if BBN_CACHE["infer"] is None:
        raise HTTPException(status_code=400, detail="Recommender not trained. Ask admin to call /api/admin/recommender/retrain")

    prefs = await _build_user_prefs(current_user["id"])

    favs = await db.favorites.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(5000)
    fav_prop_ids = set([f["property_id"] for f in favs])

    candidates = await db.properties.find({"status": "available"}, {"_id": 0}).to_list(5000)

    random.seed(7)
    if len(candidates) > 300:
        candidates = random.sample(candidates, 300)

    infer = BBN_CACHE["infer"]
    scored = []

    for p in candidates:
        if p["id"] in fav_prop_ids:
            continue

        evidence = {
            "pref_city": prefs["pref_city"],
            "pref_type": prefs["pref_type"],
            "pref_price_bucket": prefs["pref_price_bucket"],
            "prop_city": str(p.get("city_id", "unknown")),
            "prop_type": str(p.get("property_type_id", "unknown")),
            "price_bucket": _bucket_price(float(p.get("price", 0))),
            "beds_bucket": _bucket_beds(int(p.get("bedrooms", 0))),
        }

        q = infer.query(variables=[BBN_TARGET], evidence=evidence, show_progress=False)
        score = _prob_one(q)
        scored.append({"property": p, "score": round(float(score), 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: max(1, min(limit, 30))]

    return {
        "trained_at": BBN_CACHE["trained_at"],
        "rows_used": BBN_CACHE["rows_used"],
        "user_id": current_user["id"],
        "prefs": prefs,
        "recommendations": [{"property_id": x["property"]["id"], "score": x["score"], "property": x["property"]} for x in top],
    }

# ---------------- ADMIN ROUTES ----------------

@api_router.get("/admin/users", response_model=List[User])
async def get_all_users(current_user: dict = Depends(get_admin_user)):
    return await db.users.find({}, {"_id": 0, "password": 0}).to_list(1000)

@api_router.get("/admin/stats")
async def get_stats(current_user: dict = Depends(get_admin_user)):
    total_users = await db.users.count_documents({})
    total_properties = await db.properties.count_documents({})
    total_bookings = await db.bookings.count_documents({})
    pending_bookings = await db.bookings.count_documents({"status": "pending"})
    total_payments = await db.payments.count_documents({})
    total_reviews = await db.reviews.count_documents({})
    total_favorites = await db.favorites.count_documents({})

    return {
        "total_users": total_users,
        "total_properties": total_properties,
        "total_bookings": total_bookings,
        "pending_bookings": pending_bookings,
        "total_payments": total_payments,
        "total_reviews": total_reviews,
        "total_favorites": total_favorites,
    }

@api_router.get("/admin/recommender/status")
async def recommender_status(current_user: dict = Depends(get_admin_user)):
    return {
        "trained_at": BBN_CACHE["trained_at"],
        "rows_used": BBN_CACHE["rows_used"],
        "trained": BBN_CACHE["infer"] is not None,
    }

@api_router.post("/admin/recommender/retrain")
async def retrain_recommender(current_user: dict = Depends(get_admin_user)):
    result = await _train_bbn_from_db(max_pos_per_user=10, neg_ratio=3)

    BBN_CACHE["infer"] = result["infer"]
    BBN_CACHE["model"] = result["model"]
    BBN_CACHE["trained_at"] = datetime.now(timezone.utc).isoformat()
    BBN_CACHE["rows_used"] = result["rows"]

    await save_bbn_status()

    return {
        "message": "BBN recommender trained",
        "trained_at": BBN_CACHE["trained_at"],
        "rows_used": BBN_CACHE["rows_used"],
    }

@api_router.post("/admin/seed")
async def seed_data():
    admin = await db.users.find_one({"email": "admin@realestate.com"})
    if not admin:
        admin_id = str(uuid.uuid4())
        admin_doc = {
            "id": admin_id,
            "email": "admin@realestate.com",
            "password": hash_password("admin123"),
            "name": "Admin User",
            "phone": "+1234567890",
            "role": UserRole.admin.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.users.insert_one(admin_doc)
    else:
        admin_id = admin["id"]

    owner = await db.users.find_one({"email": "owner@realestate.com"})
    if not owner:
        owner_id = str(uuid.uuid4())
        owner_doc = {
            "id": owner_id,
            "email": "owner@realestate.com",
            "password": hash_password("owner123"),
            "name": "Sample Owner",
            "phone": "+9876543210",
            "role": UserRole.owner.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await db.users.insert_one(owner_doc)
    else:
        owner_id = owner["id"]

    existing_cities = await db.cities.count_documents({})
    if existing_cities == 0:
        city_docs = [
            {"id": str(uuid.uuid4()), "city_name": "Malibu"},
            {"id": str(uuid.uuid4()), "city_name": "New York"},
            {"id": str(uuid.uuid4()), "city_name": "Austin"},
            {"id": str(uuid.uuid4()), "city_name": "Chicago"},
            {"id": str(uuid.uuid4()), "city_name": "Miami"},
            {"id": str(uuid.uuid4()), "city_name": "Aspen"},
        ]
        await db.cities.insert_many(city_docs)
    else:
        city_docs = await db.cities.find({}, {"_id": 0}).to_list(100)

    existing_types = await db.property_types.count_documents({})
    if existing_types == 0:
        type_docs = [
            {"id": str(uuid.uuid4()), "type_name": "Villa"},
            {"id": str(uuid.uuid4()), "type_name": "Apartment"},
            {"id": str(uuid.uuid4()), "type_name": "House"},
            {"id": str(uuid.uuid4()), "type_name": "Penthouse"},
            {"id": str(uuid.uuid4()), "type_name": "Condo"},
            {"id": str(uuid.uuid4()), "type_name": "Cabin"},
        ]
        await db.property_types.insert_many(type_docs)
    else:
        type_docs = await db.property_types.find({}, {"_id": 0}).to_list(100)

    def find_city(name: str) -> Optional[str]:
        for c in city_docs:
            if c["city_name"] == name:
                return c["id"]
        return None

    def find_type(name: str) -> Optional[str]:
        for t in type_docs:
            if t["type_name"] == name:
                return t["id"]
        return None

    existing_props = await db.properties.count_documents({})
    if existing_props == 0:
        sample_properties = [
            {
                "id": str(uuid.uuid4()),
                "title": "Luxury Villa with Ocean View",
                "description": "Stunning 4-bedroom villa with panoramic ocean views, infinity pool, and modern amenities.",
                "price": 850000,
                "city_id": find_city("Malibu"),
                "property_type_id": find_type("Villa"),
                "location": "Malibu, CA",
                "address": "123 Ocean Drive, Malibu, CA 90265",
                "bedrooms": 4,
                "bathrooms": 3,
                "area": 3500,
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=800",
                "amenities": ["Pool", "Ocean View", "Garden", "Garage"],
                "owner_id": owner_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Modern Downtown Apartment",
                "description": "Stylish 2-bedroom apartment in downtown, close to shops and restaurants.",
                "price": 450000,
                "city_id": find_city("New York"),
                "property_type_id": find_type("Apartment"),
                "location": "New York, NY",
                "address": "456 Park Avenue, New York, NY 10022",
                "bedrooms": 2,
                "bathrooms": 2,
                "area": 1200,
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=800",
                "amenities": ["Gym", "Parking", "Rooftop Terrace"],
                "owner_id": owner_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Cozy Suburban House",
                "description": "Charming 3-bedroom home with backyard and garage.",
                "price": 320000,
                "city_id": find_city("Austin"),
                "property_type_id": find_type("House"),
                "location": "Austin, TX",
                "address": "789 Maple Street, Austin, TX 78701",
                "bedrooms": 3,
                "bathrooms": 2,
                "area": 2000,
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
                "amenities": ["Backyard", "Fireplace", "Garage"],
                "owner_id": owner_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        ]
        await db.properties.insert_many(sample_properties)

    return {"message": "Database seeded successfully"}

@api_router.post("/admin/seed-reco")
async def seed_reco_data(current_user: dict = Depends(get_admin_user)):
    cities = await db.cities.find({}, {"_id": 0}).to_list(1000)
    types = await db.property_types.find({}, {"_id": 0}).to_list(1000)
    if not cities or not types:
        raise HTTPException(status_code=400, detail="Run /api/admin/seed first to create cities/types")

    owner = await db.users.find_one({"role": UserRole.owner.value}, {"_id": 0})
    if not owner:
        raise HTTPException(status_code=400, detail="No owner found. Run /api/admin/seed first")

    TARGET_BUYERS = 30
    existing_buyers = await db.users.count_documents({"role": UserRole.buyer.value})
    buyers_to_create = max(0, TARGET_BUYERS - existing_buyers)

    for i in range(buyers_to_create):
        email = f"buyer{i+1}@mail.com"
        if await db.users.find_one({"email": email}):
            continue
        await db.users.insert_one({
            "id": str(uuid.uuid4()),
            "email": email,
            "password": hash_password("buyer123"),
            "name": f"Buyer {i+1}",
            "phone": None,
            "role": UserRole.buyer.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    buyers = await db.users.find({"role": UserRole.buyer.value}, {"_id": 0}).to_list(2000)

    TARGET_PROPS = 120
    prop_count = await db.properties.count_documents({})
    to_add = max(0, TARGET_PROPS - prop_count)
    now_iso = datetime.now(timezone.utc).isoformat()

    if to_add > 0:
        bulk = []
        for i in range(to_add):
            city = random.choice(cities)
            ptype = random.choice(types)

            price = random.choice([75000, 120000, 180000, 250000, 400000, 650000, 900000])
            bedrooms = random.choice([1, 2, 3, 4, 5])
            bathrooms = random.choice([1, 2, 3, 4])
            area = random.choice([800, 1200, 1600, 2200, 3000, 4200])

            bulk.append({
                "id": str(uuid.uuid4()),
                "title": f"Seed Property {i+1} - {ptype.get('type_name','Type')}",
                "description": "Seeded property for BBN recommendation dataset",
                "price": float(price),
                "city_id": city["id"],
                "property_type_id": ptype["id"],
                "location": f"{city.get('city_name','')}",
                "address": f"Street {random.randint(1,999)}, {city.get('city_name','')}",
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "area": float(area),
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
                "amenities": random.sample(
                    ["Pool", "Gym", "Parking", "Garden", "Garage", "Fireplace", "Rooftop", "Security"],
                    k=random.randint(2, 4)
                ),
                "owner_id": owner["id"],
                "created_at": now_iso,
            })

        await db.properties.insert_many(bulk)

    properties = await db.properties.find({}, {"_id": 0}).to_list(5000)

    created_favs = 0
    created_bookings = 0

    for b in buyers:
        fav_targets = random.sample(properties, k=min(len(properties), random.randint(8, 15)))
        for p in fav_targets:
            exists = await db.favorites.find_one({"user_id": b["id"], "property_id": p["id"]})
            if exists:
                continue
            await db.favorites.insert_one({
                "id": str(uuid.uuid4()),
                "user_id": b["id"],
                "property_id": p["id"],
                "created_at": now_iso,
            })
            created_favs += 1

        book_targets = random.sample(properties, k=min(len(properties), random.randint(1, 3)))
        for p in book_targets:
            await db.bookings.insert_one({
                "id": str(uuid.uuid4()),
                "property_id": p["id"],
                "user_id": b["id"],
                "user_name": b.get("name", ""),
                "user_email": b.get("email", ""),
                "check_in": now_iso,
                "check_out": None,
                "guests": random.randint(1, 6),
                "message": None,
                "status": "pending",
                "total_amount": float(p.get("price", 0)) * 0.01,
                "created_at": now_iso,
            })
            created_bookings += 1

    return {
        "message": "seed-reco completed",
        "buyers": len(buyers),
        "properties": len(properties),
        "favorites_created": created_favs,
        "bookings_created": created_bookings,
    }

# ---------------- UPLOADS ----------------

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@api_router.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = UPLOAD_DIR / file_name
        with open(file_path, "wb") as f:
            f.write(await file.read())

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/uploads/{file_name}"
        return {"url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed:{e}")

# ---------------- REGISTER ROUTER ----------------

app.include_router(api_router)

# ---------------- SHUTDOWN DB ----------------

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# ---------------- LOGGING ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def auto_train_recommender():
    try:
        result = await _train_bbn_from_db(max_pos_per_user=10, neg_ratio=3)

        BBN_CACHE["infer"] = result["infer"]
        BBN_CACHE["model"] = result["model"]
        BBN_CACHE["trained_at"] = datetime.now(timezone.utc).isoformat()
        BBN_CACHE["rows_used"] = result["rows"]

        logger.info("✅ Recommender auto-trained on startup")

    except Exception as e:
        logger.error(f"❌ Recommender auto-training failed: {e}")
