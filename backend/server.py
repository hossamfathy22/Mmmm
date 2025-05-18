from fastapi import FastAPI, APIRouter, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
import logging
import uuid
from pathlib import Path
from enum import Enum
import random
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import polyline

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'mandoob_db')]

# Create the main app without a prefix
app = FastAPI(title="Mandoob Pro API", 
             description="API for delivery driver assistance app",
             version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Enums
class OrderStatus(str, Enum):
    PENDING = "pending"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class DeliveryApp(str, Enum):
    TALABAT = "talabat"
    UBER_EATS = "uber_eats"
    ELMENUS = "elmenus"
    OTLOB = "otlob"
    INSTASHOP = "instashop"
    CAREEM = "careem"
    BOSTA = "bosta"
    OTHER = "other"

# Models
class Location(BaseModel):
    lat: float
    lng: float
    address: Optional[str] = None

class Order(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_app: DeliveryApp
    pickup_location: Location
    delivery_location: Location
    pickup_time: datetime = Field(default_factory=datetime.utcnow)
    delivery_deadline: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    customer_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    amount: Optional[float] = None
    items: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class OrderCreate(BaseModel):
    source_app: DeliveryApp
    pickup_location: Location
    delivery_location: Location
    pickup_time: Optional[datetime] = None
    delivery_deadline: Optional[datetime] = None
    customer_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    amount: Optional[float] = None
    items: Optional[str] = None

class RoutePoint(BaseModel):
    order_id: str
    location: Location
    is_pickup: bool
    eta: Optional[datetime] = None

class Route(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    driver_id: Optional[str] = None
    points: List[RoutePoint]
    total_distance: float
    total_time: int  # in minutes
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RouteOptimizationRequest(BaseModel):
    order_ids: List[str]
    current_location: Optional[Location] = None

class RouteOptimizationResponse(BaseModel):
    route: Route
    estimated_profit: float
    individual_profit: float
    merged_profit: float
    time_saved: int  # in minutes
    distance_saved: float  # in kilometers

class NotificationParseRequest(BaseModel):
    notification_text: str
    app_name: Optional[DeliveryApp] = None

# Utility functions for route optimization
def calculate_distance(point1: Location, point2: Location) -> float:
    """Calculate Euclidean distance between two points (simplified)"""
    return distance.euclidean(
        [point1.lat, point1.lng],
        [point2.lat, point2.lng]
    ) * 111  # Rough conversion to kilometers

def optimize_route(orders: List[Order], start_location: Optional[Location] = None) -> Route:
    """Simple route optimization algorithm"""
    if not orders:
        raise ValueError("No orders to optimize")
    
    if start_location is None:
        # Use the first order's pickup location as starting point
        start_location = orders[0].pickup_location
    
    # Create a list of all points (pickup and delivery)
    all_points = []
    for order in orders:
        all_points.append({"order_id": order.id, "location": order.pickup_location, "is_pickup": True})
        all_points.append({"order_id": order.id, "location": order.delivery_location, "is_pickup": False})
    
    # Ensure pickups happen before their corresponding deliveries
    optimized_points = []
    current_location = start_location
    unvisited = all_points.copy()
    pickup_completed = set()
    
    # First, handle all pickups using a greedy approach
    while any(point["is_pickup"] for point in unvisited):
        pickup_points = [point for point in unvisited if point["is_pickup"]]
        if not pickup_points:
            break
            
        # Find nearest pickup point
        nearest = min(
            pickup_points,
            key=lambda p: calculate_distance(current_location, p["location"])
        )
        
        # Add to route
        optimized_points.append(nearest)
        pickup_completed.add(nearest["order_id"])
        unvisited.remove(nearest)
        current_location = nearest["location"]
    
    # Then, handle all deliveries
    while unvisited:
        delivery_points = [point for point in unvisited 
                          if (not point["is_pickup"] and point["order_id"] in pickup_completed)]
        if not delivery_points:
            break
            
        # Find nearest delivery point
        nearest = min(
            delivery_points,
            key=lambda p: calculate_distance(current_location, p["location"])
        )
        
        # Add to route
        optimized_points.append(nearest)
        unvisited.remove(nearest)
        current_location = nearest["location"]
    
    # Calculate total distance and time
    total_distance = 0
    prev_location = start_location
    
    for point in optimized_points:
        dist = calculate_distance(prev_location, point["location"])
        total_distance += dist
        prev_location = point["location"]
    
    # Estimate time based on average speed of 30 km/h
    total_time = int(total_distance / 30 * 60)  # Convert to minutes
    
    # Create RoutePoints for the response
    route_points = [
        RoutePoint(
            order_id=p["order_id"],
            location=p["location"],
            is_pickup=p["is_pickup"]
        ) for p in optimized_points
    ]
    
    return Route(
        points=route_points,
        total_distance=total_distance,
        total_time=total_time
    )

def calculate_profits(orders: List[Order], optimized_route: Route) -> dict:
    """Calculate individual vs merged profits"""
    # For demonstration, use a simple model
    # Individual: base fee + distance fee
    # Merged: same total base fee, but reduced total distance fee
    
    base_fee_per_order = 15  # Base fee in Egyptian Pounds
    distance_fee_per_km = 2  # Fee per kilometer in Egyptian Pounds
    
    # Calculate individual profit
    individual_distance = 0
    for order in orders:
        # Simple there and back for each order individually
        individual_distance += (
            calculate_distance(order.pickup_location, order.delivery_location) * 2
        )
    
    individual_profit = len(orders) * base_fee_per_order + individual_distance * distance_fee_per_km
    
    # Calculate merged profit
    merged_profit = len(orders) * base_fee_per_order + optimized_route.total_distance * distance_fee_per_km
    
    # Calculate time and distance saved
    time_saved = int((individual_distance / 30 * 60) - optimized_route.total_time)
    distance_saved = individual_distance - optimized_route.total_distance
    
    return {
        "individual_profit": individual_profit,
        "merged_profit": merged_profit,
        "time_saved": time_saved,
        "distance_saved": distance_saved
    }

def extract_order_from_notification(notification_text: str, app_name: Optional[DeliveryApp] = None) -> Optional[OrderCreate]:
    """Extract order details from notification text using simple pattern matching"""
    # This is a simplified mock implementation
    # In a real app, this would use more sophisticated NLP techniques
    
    # If no app specified, try to detect it
    if app_name is None:
        if "طلبات" in notification_text or "Talabat" in notification_text:
            app_name = DeliveryApp.TALABAT
        elif "Uber" in notification_text:
            app_name = DeliveryApp.UBER_EATS
        elif "Elmenus" in notification_text:
            app_name = DeliveryApp.ELMENUS
        elif "Otlob" in notification_text:
            app_name = DeliveryApp.OTLOB
        else:
            app_name = DeliveryApp.OTHER
    
    # Generate mock data for testing
    # In a real implementation, this would parse the actual notification text
    mock_locations = [
        {"lat": 30.0444, "lng": 31.2357, "address": "Downtown Cairo"},
        {"lat": 30.0566, "lng": 31.2394, "address": "Tahrir Square"},
        {"lat": 30.0455, "lng": 31.2240, "address": "Cairo University"},
        {"lat": 30.0626, "lng": 31.2497, "address": "Ramses Square"},
        {"lat": 30.0700, "lng": 31.2200, "address": "Dokki"},
    ]
    
    pickup = random.choice(mock_locations)
    delivery = random.choice([loc for loc in mock_locations if loc != pickup])
    
    # Try to extract restaurant and customer names (simplified)
    restaurant_name = None
    customer_name = None
    
    if "restaurant" in notification_text:
        restaurant_name = notification_text.split("restaurant")[1].split(",")[0].strip()
    elif "مطعم" in notification_text:
        restaurant_name = notification_text.split("مطعم")[1].split("،")[0].strip()
    
    if "customer" in notification_text:
        customer_name = notification_text.split("customer")[1].split(",")[0].strip()
    elif "عميل" in notification_text:
        customer_name = notification_text.split("عميل")[1].split("،")[0].strip()
    
    # Create mock order with some randomization for testing
    return OrderCreate(
        source_app=app_name,
        pickup_location=Location(lat=pickup["lat"], lng=pickup["lng"], address=pickup["address"]),
        delivery_location=Location(lat=delivery["lat"], lng=delivery["lng"], address=delivery["address"]),
        restaurant_name=restaurant_name or f"Restaurant {random.randint(1, 100)}",
        customer_name=customer_name or f"Customer {random.randint(1, 100)}",
        amount=random.randint(50, 500)
    )

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Welcome to Mandoob Pro API"}

@api_router.post("/orders", response_model=Order)
async def create_order(order: OrderCreate):
    order_dict = order.model_dump()
    order_obj = Order(**order_dict)
    await db.orders.insert_one(order_obj.model_dump())
    return order_obj

@api_router.get("/orders", response_model=List[Order])
async def get_orders(status: Optional[OrderStatus] = None, limit: int = 50):
    """Get orders with optional status filter"""
    query = {}
    if status:
        query["status"] = status
    
    orders = await db.orders.find(query).sort("created_at", -1).limit(limit).to_list(limit)
    return [Order(**order) for order in orders]

@api_router.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str):
    """Get order by ID"""
    order = await db.orders.find_one({"id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return Order(**order)

@api_router.put("/orders/{order_id}", response_model=Order)
async def update_order(order_id: str, order_update: dict = Body(...)):
    """Update order fields"""
    order = await db.orders.find_one({"id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Update only provided fields
    update_data = {k: v for k, v in order_update.items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()
    
    await db.orders.update_one({"id": order_id}, {"$set": update_data})
    updated_order = await db.orders.find_one({"id": order_id})
    return Order(**updated_order)

@api_router.post("/orders/optimize", response_model=RouteOptimizationResponse)
async def optimize_orders(request: RouteOptimizationRequest):
    """Optimize route for multiple orders"""
    # Fetch orders from database
    orders = []
    for order_id in request.order_ids:
        order_data = await db.orders.find_one({"id": order_id})
        if not order_data:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        orders.append(Order(**order_data))
    
    # Perform route optimization
    optimized_route = optimize_route(orders, request.current_location)
    
    # Calculate profits
    profit_info = calculate_profits(orders, optimized_route)
    
    return RouteOptimizationResponse(
        route=optimized_route,
        estimated_profit=profit_info["merged_profit"],
        individual_profit=profit_info["individual_profit"],
        merged_profit=profit_info["merged_profit"],
        time_saved=profit_info["time_saved"],
        distance_saved=profit_info["distance_saved"]
    )

@api_router.post("/notification/parse", response_model=OrderCreate)
async def parse_notification(request: NotificationParseRequest):
    """Parse notification text to extract order details"""
    order = extract_order_from_notification(request.notification_text, request.app_name)
    if not order:
        raise HTTPException(status_code=400, detail="Could not extract order details from notification")
    return order

@api_router.get("/mock/orders", response_model=List[Order])
async def get_mock_orders(count: int = Query(5, ge=1, le=20)):
    """Generate mock orders for testing"""
    app_options = list(DeliveryApp)
    
    # Generate random locations in Cairo
    locations = [
        {"lat": 30.0444, "lng": 31.2357, "address": "Downtown Cairo"},
        {"lat": 30.0566, "lng": 31.2394, "address": "Tahrir Square"},
        {"lat": 30.0455, "lng": 31.2240, "address": "Cairo University"},
        {"lat": 30.0626, "lng": 31.2497, "address": "Ramses Square"},
        {"lat": 30.0700, "lng": 31.2200, "address": "Dokki"},
        {"lat": 30.0571, "lng": 31.2272, "address": "Mohandiseen"},
        {"lat": 30.0484, "lng": 31.2354, "address": "Garden City"},
        {"lat": 30.0751, "lng": 31.2394, "address": "Heliopolis"},
    ]
    
    # Generate mock orders
    mock_orders = []
    for i in range(count):
        pickup_idx = random.randint(0, len(locations) - 1)
        delivery_idx = random.randint(0, len(locations) - 1)
        while delivery_idx == pickup_idx:
            delivery_idx = random.randint(0, len(locations) - 1)
        
        pickup = locations[pickup_idx]
        delivery = locations[delivery_idx]
        
        # Create mock order
        order = Order(
            id=str(uuid.uuid4()),
            source_app=random.choice(app_options),
            pickup_location=Location(lat=pickup["lat"], lng=pickup["lng"], address=pickup["address"]),
            delivery_location=Location(lat=delivery["lat"], lng=delivery["lng"], address=delivery["address"]),
            pickup_time=datetime.utcnow(),
            delivery_deadline=None,
            status=OrderStatus.PENDING,
            restaurant_name=f"Restaurant {random.randint(1, 100)}",
            customer_name=f"Customer {random.randint(1, 100)}",
            amount=random.randint(50, 500),
            items=f"Order Items {random.randint(1, 100)}"
        )
        mock_orders.append(order)
    
    return mock_orders

@api_router.get("/mock/optimize", response_model=RouteOptimizationResponse)
async def get_mock_optimization():
    """Generate mock optimization response for testing"""
    # Get 3 mock orders
    mock_orders = await get_mock_orders(3)
    
    # Generate a mock starting location (driver location)
    current_location = Location(
        lat=30.0600, 
        lng=31.2300,
        address="Current Location"
    )
    
    # Optimize the route
    optimized_route = optimize_route(mock_orders, current_location)
    
    # Calculate profits
    profit_info = calculate_profits(mock_orders, optimized_route)
    
    return RouteOptimizationResponse(
        route=optimized_route,
        estimated_profit=profit_info["merged_profit"],
        individual_profit=profit_info["individual_profit"],
        merged_profit=profit_info["merged_profit"],
        time_saved=profit_info["time_saved"],
        distance_saved=profit_info["distance_saved"]
    )

# Include the router in the main app
app.include_router(api_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup():
    logger.info("Mandoob Pro API starting up")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    logger.info("Mandoob Pro API shutting down")
