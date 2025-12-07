from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import jwt
import os
import logging
import uuid
from pathlib import Path
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles

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
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
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

allowed_origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROLES & DEPENDENCIES ----------------


class UserRole(str, Enum):
    buyer = "buyer"
    owner = "owner"  # dealer
    admin = "admin"


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        )

    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != UserRole.admin.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ---------------- Pydantic MODELS ----------------

# ---- Auth / User ----


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None
    role: UserRole = UserRole.buyer  # buyer / owner / admin (admin used from seed only)


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


# ---- Cities & Property Types ----


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


# ---- Properties ----


class PropertyCreate(BaseModel):
    title: str
    description: str
    price: float
    # For DBMS spec, we have city & property type as separate tables:
    city_id: Optional[str] = None
    property_type_id: Optional[str] = None
    # For frontend compatibility, keep location/address too:
    location: str
    address: str
    bedrooms: int
    bathrooms: int
    area: float
    status: str = "available"  # available / booked / sold
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


# ---- Bookings ----


class BookingCreate(BaseModel):
    property_id: str
    check_in: str  # maps to start_date in DBMS concept
    check_out: Optional[str] = None  # maps to end_date (optional)
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
    status: str = "pending"  # pending / confirmed / cancelled
    total_amount: float
    created_at: str


# ---- Payments ----


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
    status: str  # paid / pending / failed
    created_at: str


# ---- Reviews ----


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


# ---- Favorites (Wishlist) ----


class FavoriteCreate(BaseModel):
    property_id: str


class Favorite(BaseModel):
    id: str
    user_id: str
    property_id: str
    created_at: str


# ---------------- AUTH ROUTES ----------------


@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    # Check existing email
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

    access_token = create_access_token(
        {"user_id": user_id, "email": user_data.email, "role": user_data.role.value}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name,
            "role": user_data.role.value,
        },
    }


@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(
        {"user_id": user["id"], "email": user["email"], "role": user["role"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        },
    }


@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "role": current_user["role"],
    }


# ---------------- CITY & PROPERTY TYPE ROUTES ----------------

@api_router.get("/cities", response_model=List[City])
async def list_cities():
    docs = await db.cities.find({}, {"_id": 0}).to_list(1000)
    return docs


@api_router.post("/cities", response_model=City)
async def create_city(
    city_data: CityCreate, current_user: dict = Depends(get_admin_user)
):
    city_id = str(uuid.uuid4())
    doc = {
        "id": city_id,
        "city_name": city_data.city_name,
    }
    await db.cities.insert_one(doc)
    return doc


@api_router.get("/property-types", response_model=List[PropertyType])
async def list_property_types():
    docs = await db.property_types.find({}, {"_id": 0}).to_list(1000)
    return docs


@api_router.post("/property-types", response_model=PropertyType)
async def create_property_type(
    type_data: PropertyTypeCreate, current_user: dict = Depends(get_admin_user)
):
    type_id = str(uuid.uuid4())
    doc = {
        "id": type_id,
        "type_name": type_data.type_name,
    }
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
        price_filter = {}
        if min_price is not None:
            price_filter["$gte"] = min_price
        if max_price is not None:
            price_filter["$lte"] = max_price
        query["price"] = price_filter

    properties = await db.properties.find(query, {"_id": 0}).to_list(1000)
    return properties


@api_router.get("/properties/{property_id}", response_model=Property)
async def get_property(property_id: str):
    doc = await db.properties.find_one({"id": property_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Property not found")
    return doc


@api_router.post("/properties", response_model=Property)
async def create_property(
    property_data: PropertyCreate, current_user: dict = Depends(get_current_user)
):
    if current_user["role"] not in [UserRole.owner.value, UserRole.admin.value]:
        raise HTTPException(
            status_code=403,
            detail="Only owners (dealers) or admin can create properties",
        )

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
async def update_property(
    property_id: str,
    property_data: PropertyCreate,
    current_user: dict = Depends(get_current_user),
):
    existing = await db.properties.find_one({"id": property_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Property not found")

    if existing["owner_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this property"
        )

    updated_doc = {
        **property_data.model_dump(),
        "owner_id": existing["owner_id"],
        "created_at": existing["created_at"],
    }

    await db.properties.update_one({"id": property_id}, {"$set": updated_doc})
    final_doc = await db.properties.find_one({"id": property_id}, {"_id": 0})
    return final_doc


@api_router.delete("/properties/{property_id}")
async def delete_property(
    property_id: str, current_user: dict = Depends(get_current_user)
):
    existing = await db.properties.find_one({"id": property_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Property not found")

    if existing["owner_id"] != current_user["id"] and current_user["role"] != UserRole.admin.value:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this property"
        )

    await db.properties.delete_one({"id": property_id})
    return {"message": "Property deleted successfully"}


@api_router.get("/owner/my-properties", response_model=List[Property])
async def get_my_properties(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in [UserRole.owner.value, UserRole.admin.value]:
        raise HTTPException(
            status_code=403, detail="Only owners or admin can view this"
        )

    query = {}
    if current_user["role"] == UserRole.owner.value:
        query["owner_id"] = current_user["id"]

    docs = await db.properties.find(query, {"_id": 0}).to_list(1000)
    return docs


# ---------------- BOOKING ROUTES ----------------

@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(current_user: dict = Depends(get_current_user)):
    # buyer → their bookings
    # owner → bookings for their properties
    # admin → all bookings
    if current_user["role"] == UserRole.admin.value:
        docs = await db.bookings.find({}, {"_id": 0}).to_list(1000)
        return docs

    if current_user["role"] == UserRole.owner.value:
        # find properties owned by this owner
        owner_props = await db.properties.find(
            {"owner_id": current_user["id"]}, {"id": 1}
        ).to_list(1000)
        prop_ids = [p["id"] for p in owner_props]
        docs = await db.bookings.find(
            {"property_id": {"$in": prop_ids}}, {"_id": 0}
        ).to_list(1000)
        return docs

    # buyer
    docs = await db.bookings.find(
        {"user_id": current_user["id"]}, {"_id": 0}
    ).to_list(1000)
    return docs


@api_router.get("/bookings/{booking_id}", response_model=Booking)
async def get_booking(
    booking_id: str, current_user: dict = Depends(get_current_user)
):
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if (
        booking["user_id"] != current_user["id"]
        and current_user["role"] != UserRole.admin.value
    ):
        # Owner view: check if property belongs to owner
        prop = await db.properties.find_one({"id": booking["property_id"]})
        if not prop or prop["owner_id"] != current_user["id"]:
            raise HTTPException(
                status_code=403, detail="Not authorized to view this booking"
            )

    return booking


@api_router.post("/bookings", response_model=Booking)
async def create_booking(
    booking_data: BookingCreate, current_user: dict = Depends(get_current_user)
):
    if current_user["role"] not in [UserRole.buyer.value, UserRole.admin.value]:
        raise HTTPException(
            status_code=403,
            detail="Only buyers or admin can create bookings",
        )

    prop = await db.properties.find_one({"id": booking_data.property_id})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    if prop["status"] != "available":
        raise HTTPException(
            status_code=400, detail="Property is not available for booking"
        )

    # Prevent booking own property (if buyer is also owner)
    if prop["owner_id"] == current_user["id"]:
        raise HTTPException(
            status_code=400, detail="You cannot book your own property"
        )

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
async def update_booking_status(
    booking_id: str,
    status: str,
    current_user: dict = Depends(get_current_user),
):
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    prop = await db.properties.find_one({"id": booking["property_id"]})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    # Only property owner or admin can change status
    if (
        current_user["role"] != UserRole.admin.value
        and prop["owner_id"] != current_user["id"]
    ):
        raise HTTPException(
            status_code=403,
            detail="Only property owner or admin can update booking status",
        )

    await db.bookings.update_one(
        {"id": booking_id}, {"$set": {"status": status}}
    )

    # Optionally mark property as booked if status confirmed
    if status == "confirmed":
        await db.properties.update_one(
            {"id": booking["property_id"]}, {"$set": {"status": "booked"}}
        )

    updated = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    return updated


@api_router.delete("/bookings/{booking_id}")
async def delete_booking(
    booking_id: str, current_user: dict = Depends(get_current_user)
):
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    if (
        booking["user_id"] != current_user["id"]
        and current_user["role"] != UserRole.admin.value
    ):
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this booking"
        )

    await db.bookings.delete_one({"id": booking_id})
    return {"message": "Booking deleted successfully"}


# ---------------- PAYMENT ROUTES ----------------

@api_router.get("/payments", response_model=List[Payment])
async def list_payments(current_user: dict = Depends(get_current_user)):
    if current_user["role"] == UserRole.admin.value:
        docs = await db.payments.find({}, {"_id": 0}).to_list(1000)
        return docs

    # buyer → payments for their bookings
    bookings = await db.bookings.find(
        {"user_id": current_user["id"]}, {"id": 1}
    ).to_list(1000)
    booking_ids = [b["id"] for b in bookings]

    docs = await db.payments.find(
        {"booking_id": {"$in": booking_ids}}, {"_id": 0}
    ).to_list(1000)
    return docs


@api_router.post("/payments", response_model=Payment)
async def create_payment(
    payment_data: PaymentCreate, current_user: dict = Depends(get_current_user)
):
    booking = await db.bookings.find_one({"id": payment_data.booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")

    # Only booking owner (buyer) or admin can create payment
    if (
        booking["user_id"] != current_user["id"]
        and current_user["role"] != UserRole.admin.value
    ):
        raise HTTPException(
            status_code=403,
            detail="Not authorized to create payment for this booking",
        )

    pay_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    doc = {
        "id": pay_id,
        "booking_id": payment_data.booking_id,
        "amount": payment_data.amount,
        "method": payment_data.method.value,
        "payment_date": now_iso,
        "status": "paid",  # simple: assume success
        "created_at": now_iso,
    }
    await db.payments.insert_one(doc)

    # Mark booking as confirmed if payment successful
    await db.bookings.update_one(
        {"id": payment_data.booking_id},
        {"$set": {"status": "confirmed"}},
    )

    return doc


# ---------------- REVIEW ROUTES ----------------

@api_router.get("/properties/{property_id}/reviews", response_model=List[Review])
async def list_reviews_for_property(property_id: str):
    docs = await db.reviews.find(
        {"property_id": property_id}, {"_id": 0}
    ).to_list(1000)
    return docs


@api_router.post("/reviews", response_model=Review)
async def create_review(
    review_data: ReviewCreate, current_user: dict = Depends(get_current_user)
):
    # (Optional extra check: only allow review if user has booking)
    # For now, just require authentication.
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
    docs = await db.favorites.find(
        {"user_id": current_user["id"]}, {"_id": 0}
    ).to_list(1000)
    return docs


@api_router.post("/favorites", response_model=Favorite)
async def add_favorite(
    fav_data: FavoriteCreate, current_user: dict = Depends(get_current_user)
):
    existing = await db.favorites.find_one(
        {"user_id": current_user["id"], "property_id": fav_data.property_id}
    )
    if existing:
        raise HTTPException(
            status_code=400, detail="Property already in favorites"
        )

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
async def remove_favorite(
    favorite_id: str, current_user: dict = Depends(get_current_user)
):
    fav = await db.favorites.find_one({"id": favorite_id})
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite not found")

    if fav["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=403, detail="Not authorized to remove this favorite"
        )

    await db.favorites.delete_one({"id": favorite_id})
    return {"message": "Favorite removed successfully"}


# ---------------- ADMIN ROUTES ----------------

@api_router.get("/admin/users", response_model=List[User])
async def get_all_users(current_user: dict = Depends(get_admin_user)):
    docs = await db.users.find(
        {}, {"_id": 0, "password": 0}
    ).to_list(1000)
    return docs


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


@api_router.post("/admin/seed")
async def seed_data():
    """
    Seed:
    - admin user
    - one owner
    - some cities, property types
    - sample properties
    """
    # Admin user
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

    # Owner (dealer)
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

    # Cities
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

    # Property Types
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

    # Helper maps
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

    # Properties
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


UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@api_router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = UPLOAD_DIR / file_name

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        image_url = f"http://localhost:8000/uploads/{file_name}"

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
