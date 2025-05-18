import requests
import unittest
import json
from datetime import datetime

class MandoobProAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MandoobProAPITester, self).__init__(*args, **kwargs)
        # Use the public endpoint from frontend/.env
        self.base_url = "https://d8833ce6-be4f-47f0-8dfb-2d0e6dc79840.preview.emergentagent.com/api"
        self.test_order_id = None

    def test_01_root_endpoint(self):
        """Test the root API endpoint"""
        print("\n🔍 Testing root endpoint...")
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Welcome to Mandoob Pro API")
        print("✅ Root endpoint test passed")

    def test_02_get_mock_orders(self):
        """Test getting mock orders"""
        print("\n🔍 Testing mock orders endpoint...")
        response = requests.get(f"{self.base_url}/mock/orders?count=3")
        self.assertEqual(response.status_code, 200)
        orders = response.json()
        self.assertIsInstance(orders, list)
        self.assertEqual(len(orders), 3)
        
        # Verify order structure
        for order in orders:
            self.assertIn("id", order)
            self.assertIn("source_app", order)
            self.assertIn("pickup_location", order)
            self.assertIn("delivery_location", order)
            self.assertIn("status", order)
        
        # Save an order ID for later tests
        if orders:
            self.test_order_id = orders[0]["id"]
            
        print(f"✅ Mock orders test passed - received {len(orders)} orders")

    def test_03_get_order_by_id(self):
        """Test getting a specific order by ID"""
        if not self.test_order_id:
            self.skipTest("No order ID available from previous test")
            
        print(f"\n🔍 Testing get order by ID: {self.test_order_id}...")
        response = requests.get(f"{self.base_url}/orders/{self.test_order_id}")
        self.assertEqual(response.status_code, 200)
        order = response.json()
        self.assertEqual(order["id"], self.test_order_id)
        print("✅ Get order by ID test passed")

    def test_04_create_order(self):
        """Test creating a new order"""
        print("\n🔍 Testing order creation...")
        new_order = {
            "source_app": "talabat",
            "pickup_location": {
                "lat": 30.0444,
                "lng": 31.2357,
                "address": "Downtown Cairo"
            },
            "delivery_location": {
                "lat": 30.0566,
                "lng": 31.2394,
                "address": "Tahrir Square"
            },
            "restaurant_name": "Test Restaurant",
            "customer_name": "Test Customer",
            "amount": 150
        }
        
        response = requests.post(f"{self.base_url}/orders", json=new_order)
        self.assertEqual(response.status_code, 200)
        created_order = response.json()
        self.assertIn("id", created_order)
        self.assertEqual(created_order["restaurant_name"], "Test Restaurant")
        self.assertEqual(created_order["status"], "pending")
        
        # Save the created order ID
        self.created_order_id = created_order["id"]
        print(f"✅ Order creation test passed - ID: {self.created_order_id}")

    def test_05_update_order(self):
        """Test updating an order"""
        if not hasattr(self, 'created_order_id'):
            self.skipTest("No created order ID available from previous test")
            
        print(f"\n🔍 Testing order update for ID: {self.created_order_id}...")
        update_data = {
            "status": "picked_up",
            "customer_name": "Updated Customer"
        }
        
        response = requests.put(f"{self.base_url}/orders/{self.created_order_id}", json=update_data)
        self.assertEqual(response.status_code, 200)
        updated_order = response.json()
        self.assertEqual(updated_order["id"], self.created_order_id)
        self.assertEqual(updated_order["status"], "picked_up")
        self.assertEqual(updated_order["customer_name"], "Updated Customer")
        print("✅ Order update test passed")

    def test_06_route_optimization(self):
        """Test route optimization"""
        print("\n🔍 Testing route optimization...")
        # Get some order IDs first
        response = requests.get(f"{self.base_url}/mock/orders?count=3")
        self.assertEqual(response.status_code, 200)
        orders = response.json()
        order_ids = [order["id"] for order in orders]
        
        optimization_request = {
            "order_ids": order_ids,
            "current_location": {
                "lat": 30.0600,
                "lng": 31.2300
            }
        }
        
        response = requests.post(f"{self.base_url}/orders/optimize", json=optimization_request)
        self.assertEqual(response.status_code, 200)
        optimization = response.json()
        
        # Verify optimization response structure
        self.assertIn("route", optimization)
        self.assertIn("estimated_profit", optimization)
        self.assertIn("individual_profit", optimization)
        self.assertIn("merged_profit", optimization)
        self.assertIn("time_saved", optimization)
        self.assertIn("distance_saved", optimization)
        
        # Verify route structure
        route = optimization["route"]
        self.assertIn("points", route)
        self.assertIn("total_distance", route)
        self.assertIn("total_time", route)
        
        print("✅ Route optimization test passed")

    def test_07_notification_parsing(self):
        """Test notification parsing"""
        print("\n🔍 Testing notification parsing...")
        notification_request = {
            "notification_text": "New order from restaurant KFC for customer Ahmed",
            "app_name": "talabat"
        }
        
        response = requests.post(f"{self.base_url}/notification/parse", json=notification_request)
        self.assertEqual(response.status_code, 200)
        parsed_order = response.json()
        
        # Verify parsed order structure
        self.assertEqual(parsed_order["source_app"], "talabat")
        self.assertIn("pickup_location", parsed_order)
        self.assertIn("delivery_location", parsed_order)
        
        print("✅ Notification parsing test passed")

    def test_08_mock_optimization(self):
        """Test mock optimization endpoint"""
        print("\n🔍 Testing mock optimization endpoint...")
        response = requests.get(f"{self.base_url}/mock/optimize")
        self.assertEqual(response.status_code, 200)
        optimization = response.json()
        
        # Verify optimization response structure
        self.assertIn("route", optimization)
        self.assertIn("estimated_profit", optimization)
        self.assertIn("individual_profit", optimization)
        self.assertIn("merged_profit", optimization)
        self.assertIn("time_saved", optimization)
        self.assertIn("distance_saved", optimization)
        
        print("✅ Mock optimization test passed")

if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)