import React, { useState, useEffect, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import {
  MantineProvider,
  AppShell,
  Burger,
  Group,
  Title,
  Text,
  Card,
  Badge,
  Button,
  Avatar,
  Drawer,
  List,
  ThemeIcon,
  Box,
  Paper,
  Grid,
  Container,
  Flex,
  Progress,
  SimpleGrid,
  Modal,
  TextInput,
  Tabs,
  rem,
  Tooltip,
  ActionIcon,
  Center,
  LoadingOverlay,
  Divider
} from "@mantine/core";
import { 
  useDisclosure,
  useToggle,
  useMediaQuery
} from "@mantine/hooks";
import { notifications } from "@mantine/notifications";
import { 
  IconArrowUp, 
  IconArrowDown, 
  IconCamera, 
  IconBell, 
  IconChevronRight,
  IconGraph,
  IconMap,
  IconSettings,
  IconHome,
  IconSquareRoundedPlus,
  IconClock,
  IconCash,
  IconStars,
  IconRoute,
  IconChevronLeft,
  IconCheck,
  IconX,
  IconMapPin,
  IconDirections,
  IconAlertTriangle,
  IconSpeakerphone,
  IconStarFilled,
  IconNotes,
  IconDeviceMobile,
  IconScan
} from "@tabler/icons-react";
import { GoogleMap, useLoadScript, Marker, DirectionsRenderer } from "@react-google-maps/api";
import Lottie from "lottie-react";
import deliveryAnimation from "./assets/delivery-animation.json";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Mock Google Maps API Key (replace with real key later)
const GOOGLE_MAPS_API_KEY = "MOCK_KEY";

// App logo (SVG string)
const AppLogo = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <path d="M8 14s1.5 2 4 2 4-2 4-2" />
    <line x1="9" y1="9" x2="9.01" y2="9" />
    <line x1="15" y1="9" x2="15.01" y2="9" />
  </svg>
);

// App color theme
const theme = {
  colors: {
    primary: ["#1A237E", "#283593", "#303F9F", "#3949AB", "#3F51B5"],
    secondary: ["#FF8F00", "#FFA000", "#FFB300", "#FFC107", "#FFCA28"],
    dark: ["#2A2E43", "#373B53", "#444963", "#515673", "#5D6382"],
  },
  primaryColor: "primary",
  fontFamily: "'Cairo', 'Roboto', sans-serif",
  dir: "rtl", // Right-to-left for Arabic
  defaultRadius: "md",
  components: {
    Button: {
      defaultProps: {
        radius: "xl",
      },
    },
  },
};

// Helper function to generate mock locations in Cairo
function getRandomCairoLocations(count) {
  const cairoCenter = { lat: 30.0444, lng: 31.2357 };
  const locations = [];
  
  for (let i = 0; i < count; i++) {
    const lat = cairoCenter.lat + (Math.random() - 0.5) * 0.1;
    const lng = cairoCenter.lng + (Math.random() - 0.5) * 0.1;
    locations.push({ lat, lng });
  }
  
  return locations;
}

// Helper function to get app color based on app name
function getAppColor(appName) {
  const colorMap = {
    talabat: "#FF5A00",
    uber_eats: "#06C167",
    elmenus: "#FB5607",
    otlob: "#E2142D",
    instashop: "#FFBE0B",
    careem: "#53B175",
    bosta: "#2EC5CE",
    other: "#6C757D"
  };
  
  return colorMap[appName] || colorMap.other;
}

// Helper function to get app icon based on app name
function getAppIcon(appName) {
  switch (appName) {
    case "talabat":
      return "🍔";
    case "uber_eats":
      return "🥗";
    case "elmenus":
      return "🍕";
    case "otlob":
      return "🥡";
    case "instashop":
      return "🛒";
    case "careem":
      return "🚗";
    case "bosta":
      return "📦";
    default:
      return "📱";
  }
}

// Splash Screen Component
const SplashScreen = ({ onFinished }) => {
  const [animationError, setAnimationError] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      onFinished();
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [onFinished]);
  
  return (
    <Center style={{ height: "100vh", background: "#1A237E" }}>
      <Box style={{ textAlign: "center", color: "white" }}>
        {!animationError ? (
          <Lottie 
            animationData={deliveryAnimation} 
            style={{ width: 200, height: 200 }}
            onError={() => setAnimationError(true)}
          />
        ) : (
          <IconMap size={64} style={{ margin: "20px auto" }} stroke={1.5} />
        )}
        <Title order={2} mt="md">Mandoob Pro 2024</Title>
        <Text size="lg" mt="xs">مرحبًا، لنضاعف أرباحك اليوم!</Text>
      </Box>
    </Center>
  );
};

// Mock Map Component (to be replaced with real Google Maps later)
const MapComponent = ({ orders, selectedRoute }) => {
  const [directions, setDirections] = useState(null);
  
  // This would be connected to the real Google Maps Directions service
  // For now we'll simulate a route with a mock response
  useEffect(() => {
    if (selectedRoute && selectedRoute.route && selectedRoute.route.points) {
      // Mock directions data structure
      const mockDirections = {
        routes: [{
          legs: [{
            steps: [],
            duration: { text: `${selectedRoute.route.total_time} mins` },
            distance: { text: `${selectedRoute.route.total_distance.toFixed(1)} km` }
          }],
          overview_polyline: {
            points: ""
          }
        }]
      };
      setDirections(mockDirections);
    }
  }, [selectedRoute]);

  // For simplicity, show a static map as fallback
  return (
    <Paper shadow="md" p="md" withBorder style={{ height: "100%", minHeight: "300px", position: "relative" }}>
      <Center style={{ height: "100%" }}>
        <Box style={{ textAlign: "center" }}>
          <IconMap size={48} color="#1A237E" />
          <Text size="sm" mt="md">
            {orders && orders.length > 0 ? (
              `${orders.length} طلبات في المنطقة`
            ) : (
              "جاري تحميل الخريطة..."
            )}
          </Text>
          
          {directions && (
            <Box mt="lg">
              <Title order={5}>تفاصيل المسار:</Title>
              <Text>المسافة: {selectedRoute.route.total_distance.toFixed(1)} كم</Text>
              <Text>الوقت المتوقع: {selectedRoute.route.total_time} دقيقة</Text>
            </Box>
          )}
        </Box>
      </Center>
      
      {/* Placeholder for the actual markers (would be real map markers with Google Maps) */}
      {orders && orders.length > 0 && (
        <Box style={{ position: "absolute", bottom: "10px", left: "10px" }}>
          <Flex gap="sm">
            {orders.map((order, index) => (
              <Tooltip key={index} label={`${order.restaurant_name} - ${order.source_app}`}>
                <Box 
                  style={{ 
                    width: "12px", 
                    height: "12px", 
                    borderRadius: "50%", 
                    background: getAppColor(order.source_app)
                  }} 
                />
              </Tooltip>
            ))}
          </Flex>
        </Box>
      )}
    </Paper>
  );
};

// Order Card Component
const OrderCard = ({ order, onSelect, isSelected }) => {
  return (
    <Card 
      withBorder 
      padding="md" 
      radius="md" 
      mb="md"
      style={{ 
        borderLeft: isSelected ? `4px solid ${getAppColor(order.source_app)}` : undefined,
        background: isSelected ? "#F8F9FA" : undefined
      }}
    >
      <Flex justify="space-between" align="center">
        <Flex align="center" gap="sm">
          <Box
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "50%",
              background: getAppColor(order.source_app),
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.2rem"
            }}
          >
            {getAppIcon(order.source_app)}
          </Box>
          <Box>
            <Text weight={500}>{order.restaurant_name}</Text>
            <Text size="xs" color="dimmed">{order.delivery_location.address}</Text>
          </Box>
        </Flex>
        
        <Box>
          <Badge color={isSelected ? "green" : "blue"}>
            {isSelected ? "تم اختياره" : order.status}
          </Badge>
          <Text size="xs" mt="xs" align="center">
            <IconClock size={12} style={{ display: "inline", marginRight: "4px" }} />
            15 د
          </Text>
        </Box>
      </Flex>
      
      {!isSelected && (
        <Button 
          fullWidth 
          variant="light" 
          color="blue" 
          mt="md"
          leftSection={<IconSquareRoundedPlus size={16} />}
          onClick={() => onSelect(order)}
        >
          إضافة للدمج
        </Button>
      )}
    </Card>
  );
};

// Merge Suggestion Card Component
const MergeSuggestionCard = ({ optimizationResult, onAccept, onCancel }) => {
  if (!optimizationResult) return null;
  
  const profitIncrease = ((optimizationResult.merged_profit - optimizationResult.individual_profit) / optimizationResult.individual_profit * 100).toFixed(0);
  
  return (
    <Card withBorder shadow="md" radius="md" p="lg" mt="md">
      <Title order={4} mb="md" align="center">
        تحليل الدمج الذكي
      </Title>
      
      <Grid grow mb="lg">
        <Grid.Col span={6}>
          <Card withBorder p="md" radius="md" style={{ background: "#F8F9FA" }}>
            <Center>
              <IconCash size={24} color="#6C757D" />
            </Center>
            <Text align="center" weight={500} mt="xs">الربح الفردي</Text>
            <Text align="center" size="xl">{optimizationResult.individual_profit.toFixed(0)} ج</Text>
          </Card>
        </Grid.Col>
        
        <Grid.Col span={6}>
          <Card withBorder p="md" radius="md" style={{ background: "#E3F2FD" }}>
            <Center>
              <IconCash size={24} color="#2196F3" />
            </Center>
            <Text align="center" weight={500} mt="xs">الربح المدمج</Text>
            <Text align="center" size="xl" color="blue">{optimizationResult.merged_profit.toFixed(0)} ج</Text>
            <Badge color="green" variant="light" fullWidth mt="xs">
              +{profitIncrease}% ربح إضافي
            </Badge>
          </Card>
        </Grid.Col>
      </Grid>
      
      <Divider my="md" />
      
      <List spacing="xs" size="sm" mb="md" center>
        <List.Item 
          icon={
            <ThemeIcon color="green" size={24} radius="xl">
              <IconClock size={16} />
            </ThemeIcon>
          }
        >
          وفر {optimizationResult.time_saved} دقيقة في وقت التوصيل
        </List.Item>
        <List.Item 
          icon={
            <ThemeIcon color="blue" size={24} radius="xl">
              <IconRoute size={16} />
            </ThemeIcon>
          }
        >
          وفر {optimizationResult.distance_saved.toFixed(1)} كم في المسافة المقطوعة
        </List.Item>
      </List>
      
      <Flex gap="md" mt="lg">
        <Button 
          variant="outline" 
          color="red" 
          style={{ flex: 1 }}
          leftSection={<IconX size={18} />}
          onClick={onCancel}
        >
          إلغاء
        </Button>
        <Button 
          color="green" 
          style={{ flex: 1 }}
          leftSection={<IconCheck size={18} />}
          onClick={onAccept}
        >
          قبول الدمج
        </Button>
      </Flex>
    </Card>
  );
};

// Scanner Component
const ScannerComponent = ({ onScanComplete }) => {
  const [scanning, setScanning] = useState(false);
  
  const handleScan = () => {
    setScanning(true);
    
    // Simulate scanning process
    setTimeout(() => {
      setScanning(false);
      
      // Mock scan result
      const mockOrder = {
        source_app: "talabat",
        pickup_location: {
          lat: 30.0444,
          lng: 31.2357,
          address: "KFC, شارع التحرير"
        },
        delivery_location: {
          lat: 30.0566,
          lng: 31.2394,
          address: "برج القاهرة، الجزيرة"
        },
        restaurant_name: "KFC",
        customer_name: "محمد علي",
        amount: 120
      };
      
      onScanComplete(mockOrder);
      
      notifications.show({
        title: "تم تحليل الطلب بنجاح",
        message: "تم إضافة طلب جديد من KFC",
        color: "green"
      });
    }, 2000);
  };
  
  return (
    <Box style={{ textAlign: "center" }}>
      <Paper
        withBorder
        p="xl"
        radius="md"
        style={{
          height: "300px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background: "#F8F9FA",
          position: "relative"
        }}
      >
        <LoadingOverlay visible={scanning} overlayBlur={2} />
        
        <IconScan size={64} color={scanning ? "#1A237E" : "#6C757D"} />
        <Text mt="md" weight={500}>
          {scanning ? "جاري تحليل الطلب..." : "ضع الإشعار داخل الإطار"}
        </Text>
        
        {!scanning && (
          <Button
            mt="xl"
            color="primary"
            leftSection={<IconCamera size={18} />}
            onClick={handleScan}
          >
            التقاط صورة
          </Button>
        )}
      </Paper>
      
      <Text size="sm" mt="md" color="dimmed">
        يمكنك مسح إشعارات الطلب من Talabat, Uber Eats, Elmenus و المزيد من التطبيقات
      </Text>
    </Box>
  );
};

// Analytics Component
const AnalyticsComponent = () => {
  // Mock analytics data
  const weeklyEarnings = [120, 180, 150, 210, 160, 190, 230];
  const totalEarnings = weeklyEarnings.reduce((sum, val) => sum + val, 0);
  const ordersCompleted = 42;
  const avgTimePerOrder = 23;
  
  return (
    <Container>
      <Title order={3} mb="lg">تحليلات الأداء</Title>
      
      <Card withBorder p="lg" radius="md" mb="lg">
        <Text size="lg" weight={500} mb="md">أرباح الأسبوع (بالجنيه)</Text>
        
        <Flex justify="space-between" mb="xs">
          {weeklyEarnings.map((earning, index) => (
            <Box key={index} style={{ textAlign: "center", flex: 1 }}>
              <div 
                style={{ 
                  height: `${earning/2}px`, 
                  background: "#1A237E",
                  margin: "0 auto",
                  width: "20px",
                  borderRadius: "3px 3px 0 0"
                }} 
              />
              <Text size="xs" mt="xs">{["السبت", "الأحد", "الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة"][index]}</Text>
            </Box>
          ))}
        </Flex>
        
        <Divider my="lg" />
        
        <SimpleGrid cols={3}>
          <Box style={{ textAlign: "center" }}>
            <Text size="xl" weight={700}>{totalEarnings} ج</Text>
            <Text size="xs" color="dimmed">إجمالي الأرباح</Text>
          </Box>
          <Box style={{ textAlign: "center" }}>
            <Text size="xl" weight={700}>{ordersCompleted}</Text>
            <Text size="xs" color="dimmed">الطلبات المكتملة</Text>
          </Box>
          <Box style={{ textAlign: "center" }}>
            <Text size="xl" weight={700}>{avgTimePerOrder} د</Text>
            <Text size="xs" color="dimmed">متوسط زمن التوصيل</Text>
          </Box>
        </SimpleGrid>
      </Card>
      
      <Card withBorder p="lg" radius="md">
        <Title order={4} mb="md">إنجازاتك</Title>
        
        <SimpleGrid cols={3} spacing="lg">
          <Card withBorder radius="md" p="md" style={{ textAlign: "center" }}>
            <IconStarFilled size={32} color="#FFD700" style={{ margin: "0 auto" }} />
            <Text mt="sm" weight={500}>سائق ذهبي</Text>
            <Progress value={80} mt="xs" color="yellow" />
          </Card>
          
          <Card withBorder radius="md" p="md" style={{ textAlign: "center" }}>
            <IconSpeakerphone size={32} color="#1A237E" style={{ margin: "0 auto" }} />
            <Text mt="sm" weight={500}>دمج المحترفين</Text>
            <Progress value={60} mt="xs" color="blue" />
          </Card>
          
          <Card withBorder radius="md" p="md" style={{ textAlign: "center" }}>
            <IconAlertTriangle size={32} color="#FF5722" style={{ margin: "0 auto" }} />
            <Text mt="sm" weight={500}>سرعة التوصيل</Text>
            <Progress value={90} mt="xs" color="orange" />
          </Card>
        </SimpleGrid>
        
        <Button fullWidth variant="light" mt="lg">
          عرض جميع الإنجازات
        </Button>
      </Card>
    </Container>
  );
};

// Main Dashboard Component
const Dashboard = () => {
  const [orders, setOrders] = useState([]);
  const [selectedOrders, setSelectedOrders] = useState([]);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [activeTab, setActiveTab] = useState("orders");
  const [scannerOpen, { open: openScanner, close: closeScanner }] = useDisclosure(false);
  
  // Fetch mock orders on mount
  useEffect(() => {
    const fetchOrders = async () => {
      try {
        const response = await axios.get(`${API}/mock/orders?count=5`);
        setOrders(response.data);
      } catch (error) {
        console.error("Error fetching orders:", error);
        notifications.show({
          title: "خطأ",
          message: "فشل في تحميل الطلبات، يرجى المحاولة مرة أخرى",
          color: "red"
        });
      }
    };
    
    fetchOrders();
  }, []);
  
  // Handle selecting an order for merging
  const handleSelectOrder = (order) => {
    const isAlreadySelected = selectedOrders.some(o => o.id === order.id);
    
    if (isAlreadySelected) {
      setSelectedOrders(selectedOrders.filter(o => o.id !== order.id));
    } else {
      setSelectedOrders([...selectedOrders, order]);
      
      // Show notification for first selection
      if (selectedOrders.length === 0) {
        notifications.show({
          title: "تم إضافة الطلب",
          message: "أضف المزيد من الطلبات لتفعيل الدمج الذكي",
          color: "blue"
        });
      }
      
      // If we have at least 2 orders selected, get optimization suggestion
      if (selectedOrders.length >= 1) {
        getRouteOptimization([...selectedOrders, order]);
      }
    }
  };
  
  // Get route optimization for selected orders
  const getRouteOptimization = async (ordersToOptimize) => {
    try {
      // For the MVP, we'll use mock data
      const response = await axios.get(`${API}/mock/optimize`);
      setOptimizationResult(response.data);
    } catch (error) {
      console.error("Error optimizing route:", error);
      notifications.show({
        title: "خطأ",
        message: "فشل في حساب المسار الأمثل، يرجى المحاولة مرة أخرى",
        color: "red"
      });
    }
  };
  
  // Handle accepting the optimized route
  const handleAcceptOptimization = () => {
    notifications.show({
      title: "تم قبول المسار المُحسّن",
      message: "يمكنك الآن بدء التوصيل باتباع المسار المقترح",
      color: "green"
    });
    
    // In a real app, this would navigate to a detailed route view or open maps app
  };
  
  // Handle scan completion
  const handleScanComplete = (scannedOrder) => {
    closeScanner();
    
    // Add the scanned order to our list
    const newOrder = {
      id: `scan-${Date.now()}`,
      status: "pending",
      ...scannedOrder
    };
    
    setOrders([newOrder, ...orders]);
    
    // Auto-select it for merging
    handleSelectOrder(newOrder);
  };
  
  return (
    <Box>
      <Tabs 
        value={activeTab} 
        onChange={setActiveTab}
        variant="pills"
        radius="xl"
        mb="md"
      >
        <Tabs.List grow position="center">
          <Tabs.Tab value="orders" leftSection={<IconHome size={16} />}>
            الرئيسية
          </Tabs.Tab>
          <Tabs.Tab value="map" leftSection={<IconMap size={16} />}>
            الخريطة
          </Tabs.Tab>
          <Tabs.Tab value="analytics" leftSection={<IconGraph size={16} />}>
            التحليلات
          </Tabs.Tab>
        </Tabs.List>
      </Tabs>
      
      <Box style={{ display: activeTab === "orders" ? "block" : "none" }}>
        <Grid>
          <Grid.Col span={{ base: 12, md: 7 }}>
            <MapComponent 
              orders={orders} 
              selectedRoute={optimizationResult}
            />
            
            {optimizationResult && selectedOrders.length >= 2 && (
              <MergeSuggestionCard 
                optimizationResult={optimizationResult}
                onAccept={handleAcceptOptimization}
                onCancel={() => setOptimizationResult(null)}
              />
            )}
          </Grid.Col>
          
          <Grid.Col span={{ base: 12, md: 5 }}>
            <Flex justify="space-between" align="center" mb="md">
              <Title order={4}>الطلبات النشطة</Title>
              <Badge color="blue" size="lg">
                {orders.length} طلب
              </Badge>
            </Flex>
            
            <Box style={{ maxHeight: "500px", overflowY: "auto" }}>
              {orders.map((order) => (
                <OrderCard 
                  key={order.id}
                  order={order}
                  onSelect={handleSelectOrder}
                  isSelected={selectedOrders.some(o => o.id === order.id)}
                />
              ))}
            </Box>
          </Grid.Col>
        </Grid>
      </Box>
      
      <Box style={{ display: activeTab === "map" ? "block" : "none" }}>
        <Paper shadow="md" p="md" withBorder style={{ height: "70vh" }}>
          <MapComponent 
            orders={orders} 
            selectedRoute={optimizationResult}
          />
        </Paper>
        
        {optimizationResult && (
          <Box mt="md">
            <Button 
              fullWidth 
              color="green"
              leftSection={<IconDirections size={18} />}
            >
              فتح في تطبيق الخرائط
            </Button>
          </Box>
        )}
      </Box>
      
      <Box style={{ display: activeTab === "analytics" ? "block" : "none" }}>
        <AnalyticsComponent />
      </Box>
      
      {/* Floating action buttons */}
      <Group position="center" mt="xl" pb="xl">
        <ActionIcon 
          variant="filled" 
          color="blue" 
          size="xl" 
          radius="xl"
          onClick={openScanner}
        >
          <IconCamera size={24} />
        </ActionIcon>
        
        <ActionIcon
          variant="filled"
          color="indigo"
          size="xl"
          radius="xl"
        >
          <IconBell size={24} />
        </ActionIcon>
      </Group>
      
      {/* Scanner Modal */}
      <Modal 
        opened={scannerOpen} 
        onClose={closeScanner}
        title="مسح إشعار الطلب"
        size="lg"
        centered
      >
        <ScannerComponent onScanComplete={handleScanComplete} />
      </Modal>
    </Box>
  );
};

// Main App
function App() {
  const [showSplash, setShowSplash] = useState(true);
  
  return (
    <MantineProvider theme={theme}>
      {showSplash ? (
        <SplashScreen onFinished={() => setShowSplash(false)} />
      ) : (
        <BrowserRouter>
          <AppShell
            header={{ height: 60 }}
            padding="md"
          >
            <AppShell.Header>
              <Group h="100%" px="md" style={{ justifyContent: "space-between" }}>
                <Group>
                  <AppLogo />
                  <Title order={3}>Mandoob Pro</Title>
                </Group>
                <ActionIcon variant="subtle" color="gray">
                  <IconSettings size={20} />
                </ActionIcon>
              </Group>
            </AppShell.Header>

            <AppShell.Main>
              <Routes>
                <Route path="/" element={<Dashboard />} />
              </Routes>
            </AppShell.Main>
          </AppShell>
        </BrowserRouter>
      )}
    </MantineProvider>
  );
}

export default App;
