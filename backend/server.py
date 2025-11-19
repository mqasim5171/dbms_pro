from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import jwt

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Password hashing
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# JWT token functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Auth dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    phone: Optional[str] = None
    role: str = "user"
    created_at: str

class PropertyCreate(BaseModel):
    title: str
    description: str
    price: float
    location: str
    address: str
    bedrooms: int
    bathrooms: int
    area: float
    property_type: str
    status: str = "available"
    image_url: str
    amenities: List[str] = []

class Property(BaseModel):
    id: str
    title: str
    description: str
    price: float
    location: str
    address: str
    bedrooms: int
    bathrooms: int
    area: float
    property_type: str
    status: str
    image_url: str
    amenities: List[str]
    created_at: str
    owner_id: str

class BookingCreate(BaseModel):
    property_id: str
    check_in: str
    check_out: str
    guests: int
    message: Optional[str] = None

class Booking(BaseModel):
    id: str
    property_id: str
    user_id: str
    user_name: str
    user_email: str
    check_in: str
    check_out: str
    guests: int
    message: Optional[str] = None
    status: str = "pending"
    created_at: str

# Auth Routes
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(user_data.password)
    
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "password": hashed_password,
        "name": user_data.name,
        "phone": user_data.phone,
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)
    
    # Create token
    access_token = create_access_token({"user_id": user_id, "email": user_data.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user_data.email,
            "name": user_data.name,
            "role": "user"
        }
    }

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token({"user_id": user["id"], "email": user["email"]})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user.get("role", "user")
        }
    }

@api_router.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"],
        "role": current_user.get("role", "user")
    }

# Property Routes
@api_router.get("/properties", response_model=List[Property])
async def get_properties(status: Optional[str] = None):
    query = {}
    if status:
        query["status"] = status
    
    properties = await db.properties.find(query, {"_id": 0}).to_list(1000)
    return properties

@api_router.get("/properties/{property_id}", response_model=Property)
async def get_property(property_id: str):
    property_doc = await db.properties.find_one({"id": property_id}, {"_id": 0})
    if not property_doc:
        raise HTTPException(status_code=404, detail="Property not found")
    return property_doc

@api_router.post("/properties", response_model=Property)
async def create_property(property_data: PropertyCreate, current_user: dict = Depends(get_current_user)):
    property_id = str(uuid.uuid4())
    
    property_doc = {
        "id": property_id,
        **property_data.model_dump(),
        "owner_id": current_user["id"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.properties.insert_one(property_doc)
    return property_doc

@api_router.put("/properties/{property_id}", response_model=Property)
async def update_property(
    property_id: str,
    property_data: PropertyCreate,
    current_user: dict = Depends(get_current_user)
):
    property_doc = await db.properties.find_one({"id": property_id})
    if not property_doc:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check ownership or admin
    if property_doc["owner_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to update this property")
    
    updated_doc = {
        **property_data.model_dump(),
        "owner_id": property_doc["owner_id"],
        "created_at": property_doc["created_at"]
    }
    
    await db.properties.update_one(
        {"id": property_id},
        {"$set": updated_doc}
    )
    
    updated_property = await db.properties.find_one({"id": property_id}, {"_id": 0})
    return updated_property

@api_router.delete("/properties/{property_id}")
async def delete_property(property_id: str, current_user: dict = Depends(get_current_user)):
    property_doc = await db.properties.find_one({"id": property_id})
    if not property_doc:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check ownership or admin
    if property_doc["owner_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this property")
    
    await db.properties.delete_one({"id": property_id})
    return {"message": "Property deleted successfully"}

# Booking Routes
@api_router.get("/bookings", response_model=List[Booking])
async def get_bookings(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") == "admin":
        bookings = await db.bookings.find({}, {"_id": 0}).to_list(1000)
    else:
        bookings = await db.bookings.find({"user_id": current_user["id"]}, {"_id": 0}).to_list(1000)
    return bookings

@api_router.get("/bookings/{booking_id}", response_model=Booking)
async def get_booking(booking_id: str, current_user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Check access
    if booking["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to view this booking")
    
    return booking

@api_router.post("/bookings", response_model=Booking)
async def create_booking(booking_data: BookingCreate, current_user: dict = Depends(get_current_user)):
    # Verify property exists
    property_doc = await db.properties.find_one({"id": booking_data.property_id})
    if not property_doc:
        raise HTTPException(status_code=404, detail="Property not found")
    
    if property_doc["status"] != "available":
        raise HTTPException(status_code=400, detail="Property is not available for booking")
    
    booking_id = str(uuid.uuid4())
    
    booking_doc = {
        "id": booking_id,
        **booking_data.model_dump(),
        "user_id": current_user["id"],
        "user_name": current_user["name"],
        "user_email": current_user["email"],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.bookings.insert_one(booking_doc)
    return booking_doc

@api_router.put("/bookings/{booking_id}/status")
async def update_booking_status(
    booking_id: str,
    status: str,
    current_user: dict = Depends(get_admin_user)
):
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    await db.bookings.update_one(
        {"id": booking_id},
        {"$set": {"status": status}}
    )
    
    updated_booking = await db.bookings.find_one({"id": booking_id}, {"_id": 0})
    return updated_booking

@api_router.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, current_user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    # Check ownership or admin
    if booking["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this booking")
    
    await db.bookings.delete_one({"id": booking_id})
    return {"message": "Booking deleted successfully"}

# Admin Routes
@api_router.get("/admin/users", response_model=List[User])
async def get_all_users(current_user: dict = Depends(get_admin_user)):
    users = await db.users.find({}, {"_id": 0, "password": 0}).to_list(1000)
    return users

@api_router.get("/admin/stats")
async def get_stats(current_user: dict = Depends(get_admin_user)):
    total_users = await db.users.count_documents({})
    total_properties = await db.properties.count_documents({})
    total_bookings = await db.bookings.count_documents({})
    pending_bookings = await db.bookings.count_documents({"status": "pending"})
    
    return {
        "total_users": total_users,
        "total_properties": total_properties,
        "total_bookings": total_bookings,
        "pending_bookings": pending_bookings
    }

@api_router.post("/admin/seed")
async def seed_data():
    # Check if admin exists
    admin = await db.users.find_one({"email": "admin@realestate.com"})
    if not admin:
        admin_id = str(uuid.uuid4())
        admin_doc = {
            "id": admin_id,
            "email": "admin@realestate.com",
            "password": hash_password("admin123"),
            "name": "Admin User",
            "phone": "+1234567890",
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(admin_doc)
    
    # Check if properties exist
    property_count = await db.properties.count_documents({})
    if property_count == 0:
        sample_properties = [
            {
                "id": str(uuid.uuid4()),
                "title": "Luxury Villa with Ocean View",
                "description": "Stunning 4-bedroom villa with panoramic ocean views, infinity pool, and modern amenities. Perfect for families seeking luxury and comfort.",
                "price": 850000,
                "location": "Malibu, CA",
                "address": "123 Ocean Drive, Malibu, CA 90265",
                "bedrooms": 4,
                "bathrooms": 3,
                "area": 3500,
                "property_type": "Villa",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1613490493576-7fde63acd811?w=800",
                "amenities": ["Pool", "Ocean View", "Garden", "Garage", "Smart Home"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Modern Downtown Apartment",
                "description": "Stylish 2-bedroom apartment in the heart of downtown. Walking distance to restaurants, shops, and entertainment.",
                "price": 450000,
                "location": "New York, NY",
                "address": "456 Park Avenue, New York, NY 10022",
                "bedrooms": 2,
                "bathrooms": 2,
                "area": 1200,
                "property_type": "Apartment",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=800",
                "amenities": ["Gym", "Parking", "Concierge", "Rooftop Terrace"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Cozy Suburban House",
                "description": "Charming 3-bedroom house in a quiet neighborhood. Great schools, parks, and family-friendly community.",
                "price": 320000,
                "location": "Austin, TX",
                "address": "789 Maple Street, Austin, TX 78701",
                "bedrooms": 3,
                "bathrooms": 2,
                "area": 2000,
                "property_type": "House",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800",
                "amenities": ["Backyard", "Fireplace", "Garage", "Updated Kitchen"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Penthouse with City Skyline",
                "description": "Exclusive penthouse with breathtaking city views. Floor-to-ceiling windows, private elevator, and premium finishes.",
                "price": 1200000,
                "location": "Chicago, IL",
                "address": "321 Michigan Avenue, Chicago, IL 60601",
                "bedrooms": 3,
                "bathrooms": 3,
                "area": 2800,
                "property_type": "Penthouse",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=800",
                "amenities": ["City View", "Private Elevator", "Wine Cellar", "Smart Home", "Gym"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Beachfront Condo",
                "description": "Beautiful beachfront condo with direct beach access. Enjoy sunrise from your private balcony.",
                "price": 580000,
                "location": "Miami, FL",
                "address": "654 Ocean Boulevard, Miami, FL 33139",
                "bedrooms": 2,
                "bathrooms": 2,
                "area": 1500,
                "property_type": "Condo",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1580587771525-78b9dba3b914?w=800",
                "amenities": ["Beach Access", "Pool", "Balcony", "Security"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Mountain Retreat Cabin",
                "description": "Peaceful cabin nestled in the mountains. Perfect weekend getaway with hiking trails and nature all around.",
                "price": 275000,
                "location": "Aspen, CO",
                "address": "987 Pine Ridge Road, Aspen, CO 81611",
                "bedrooms": 2,
                "bathrooms": 1,
                "area": 1100,
                "property_type": "Cabin",
                "status": "available",
                "image_url": "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=800",
                "amenities": ["Fireplace", "Mountain View", "Deck", "Wood Stove"],
                "owner_id": admin["id"] if admin else admin_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        await db.properties.insert_many(sample_properties)
    
    return {"message": "Sample data seeded successfully"}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()