import React, { useState, useEffect } from "react";
import "@/App.css";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useParams,
} from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import {
  Home,
  Building2,
  Calendar,
  User,
  LogOut,
  Menu,
  X,
  Search,
  MapPin,
  Bed,
  Bath,
  Maximize,
  Star,
  Phone,
  Mail,
  Heart,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// ---------------- AUTH CONTEXT ----------------

const AuthContext = React.createContext();

function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("token"));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchUser();
    } else {
      setLoading(false);
    }
    // eslint-disable-next-line
  }, [token]);

  const fetchUser = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUser(response.data);
    } catch (error) {
      console.error("Failed to fetch user", error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = (newToken, userData) => {
    localStorage.setItem("token", newToken);
    setToken(newToken);
    setUser(userData);
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

const useAuth = () => React.useContext(AuthContext);

// ---------------- PROTECTED ROUTE ----------------

function ProtectedRoute({ children, adminOnly = false }) {
  const { user, loading } = useAuth();

  if (loading) return <div className="loading-screen">Loading...</div>;
  if (!user) return <Navigate to="/login" />;
  if (adminOnly && user.role !== "admin") return <Navigate to="/" />;

  return children;
}

// ---------------- NAVIGATION ----------------

function Navigation() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-brand" onClick={() => navigate("/")}>
          <Building2 className="nav-icon" />
          <span>RealtyHub</span>
        </div>

        <div className="nav-links desktop">
          <a href="/" data-testid="nav-home">
            Home
          </a>
          <a href="/properties" data-testid="nav-properties">
            Properties
          </a>

          {/* ✅ Favorites link (logged-in only) */}
          {user && (
            <a href="/favorites" data-testid="nav-favorites">
              Favorites
            </a>
          )}

          {user && (
            <a href="/bookings" data-testid="nav-bookings">
              My Bookings
            </a>
          )}
          {user?.role === "owner" && (
            <a href="/owner" data-testid="nav-owner">
              My Properties
            </a>
          )}
          {user?.role === "admin" && (
            <a href="/admin" data-testid="nav-admin">
              Admin
            </a>
          )}
        </div>

        <div className="nav-actions desktop">
          {user ? (
            <div className="user-menu">
              <span className="user-name" data-testid="user-name">
                {user.name}
              </span>
              <Button
                onClick={logout}
                variant="outline"
                size="sm"
                data-testid="logout-btn"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          ) : (
            <div className="auth-buttons">
              <Button
                onClick={() => navigate("/login")}
                variant="outline"
                data-testid="login-btn"
              >
                Login
              </Button>
              <Button
                onClick={() => navigate("/register")}
                data-testid="register-btn"
              >
                Sign Up
              </Button>
            </div>
          )}
        </div>

        <button
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          data-testid="mobile-menu-btn"
        >
          {mobileMenuOpen ? <X /> : <Menu />}
        </button>
      </div>

      {mobileMenuOpen && (
        <div className="mobile-menu" data-testid="mobile-menu">
          <a href="/" onClick={() => setMobileMenuOpen(false)}>
            Home
          </a>
          <a href="/properties" onClick={() => setMobileMenuOpen(false)}>
            Properties
          </a>

          {/* ✅ Favorites link (logged-in only) */}
          {user && (
            <a href="/favorites" onClick={() => setMobileMenuOpen(false)}>
              Favorites
            </a>
          )}

          {user && (
            <a href="/bookings" onClick={() => setMobileMenuOpen(false)}>
              My Bookings
            </a>
          )}
          {user?.role === "owner" && (
            <a href="/owner" onClick={() => setMobileMenuOpen(false)}>
              My Properties
            </a>
          )}
          {user?.role === "admin" && (
            <a href="/admin" onClick={() => setMobileMenuOpen(false)}>
              Admin
            </a>
          )}
          {user ? (
            <Button
              onClick={() => {
                logout();
                setMobileMenuOpen(false);
              }}
              variant="outline"
            >
              Logout
            </Button>
          ) : (
            <>
              <Button
                onClick={() => {
                  navigate("/login");
                  setMobileMenuOpen(false);
                }}
                variant="outline"
              >
                Login
              </Button>
              <Button
                onClick={() => {
                  navigate("/register");
                  setMobileMenuOpen(false);
                }}
              >
                Sign Up
              </Button>
            </>
          )}
        </div>
      )}
    </nav>
  );
}

// ---------------- HOME PAGE ----------------

function HomePage() {
  const navigate = useNavigate();
  const [properties, setProperties] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchProperties();
    // eslint-disable-next-line
  }, []);

  const fetchProperties = async () => {
    try {
      const response = await axios.get(`${API}/properties`);
      setProperties(response.data);
    } catch (error) {
      toast.error("Failed to load properties");
    }
  };

  const filteredProperties = properties.filter(
    (p) =>
      p.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.location.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="page-container">
      <Navigation />

      <section className="hero-section" data-testid="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">Find Your Dream Home</h1>
          <p className="hero-subtitle">
            Discover exceptional properties in prime locations
          </p>

          <div className="search-bar" data-testid="search-bar">
            <Search className="search-icon" />
            <Input
              type="text"
              placeholder="Search by location or property name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
              data-testid="search-input"
            />
            <Button data-testid="search-btn">Search</Button>
          </div>
        </div>
      </section>

      <section className="properties-section">
        <div className="section-header">
          <h2>Featured Properties</h2>
          <p>Explore our hand-picked selection of premium real estate</p>
        </div>

        <div className="properties-grid" data-testid="properties-grid">
          {filteredProperties.map((property) => (
            <Card
              key={property.id}
              className="property-card"
              data-testid={`property-card-${property.id}`}
            >
              <div
                className="property-image"
                onClick={() => navigate(`/property/${property.id}`)}
              >
                <img src={property.image_url} alt={property.title} />
                <div className="property-badge">{property.status}</div>
              </div>
              <CardContent className="property-info">
                <h3 className="property-title" data-testid="property-title">
                  {property.title}
                </h3>
                <div className="property-location">
                  <MapPin className="w-4 h-4" />
                  <span>{property.location}</span>
                </div>
                <div className="property-features">
                  <span>
                    <Bed className="w-4 h-4" /> {property.bedrooms}
                  </span>
                  <span>
                    <Bath className="w-4 h-4" /> {property.bathrooms}
                  </span>
                  <span>
                    <Maximize className="w-4 h-4" /> {property.area} sq ft
                  </span>
                </div>
                <div className="property-footer">
                  <span className="property-price" data-testid="property-price">
                    ${property.price.toLocaleString()}
                  </span>
                  <Button
                    onClick={() => navigate(`/property/${property.id}`)}
                    data-testid="view-details-btn"
                  >
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

// ---------------- FAVORITES PAGE (✅ NEW) ----------------

function FavoritesPage() {
  const { token } = useAuth();
  const navigate = useNavigate();

  const [favorites, setFavorites] = useState([]);
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFavorites();
    // eslint-disable-next-line
  }, []);

  const fetchFavorites = async () => {
    setLoading(true);
    try {
      const favRes = await axios.get(`${API}/favorites`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const favList = favRes.data || [];
      setFavorites(favList);

      const propertyPromises = favList.map((f) =>
        axios.get(`${API}/properties/${f.property_id}`).catch(() => null)
      );

      const propertyResponses = await Promise.all(propertyPromises);
      const props = propertyResponses
        .filter((r) => r?.data)
        .map((r) => r.data);

      setProperties(props);
    } catch (err) {
      toast.error("Failed to load favorites");
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (propertyId) => {
    const fav = favorites.find((f) => f.property_id === propertyId);
    if (!fav) return;

    try {
      await axios.delete(`${API}/favorites/${fav.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      toast.success("Removed from favorites");
      fetchFavorites();
    } catch (err) {
      toast.error("Failed to remove favorite");
    }
  };

  return (
    <div className="page-container">
      <Navigation />

      <div className="properties-page">
        <div className="page-header">
          <h1>My Favorites</h1>
          <p className="page-subtitle">All properties you saved</p>
        </div>

        {loading ? (
          <div className="loading-screen">Loading favorites...</div>
        ) : properties.length === 0 ? (
          <Card className="empty-state" data-testid="empty-favorites">
            <CardContent>
              <Heart className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>You haven&apos;t saved any properties yet.</p>
              <Button onClick={() => navigate("/properties")} className="mt-4">
                Browse Properties
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="properties-grid" data-testid="favorites-grid">
            {properties.map((property) => (
              <Card key={property.id} className="property-card">
                <div
                  className="property-image"
                  onClick={() => navigate(`/property/${property.id}`)}
                >
                  <img src={property.image_url} alt={property.title} />
                  <div className="property-badge">{property.status}</div>
                </div>
                <CardContent className="property-info">
                  <h3 className="property-title">{property.title}</h3>
                  <div className="property-location">
                    <MapPin className="w-4 h-4" />
                    <span>{property.location}</span>
                  </div>
                  <div className="property-features">
                    <span>
                      <Bed className="w-4 h-4" /> {property.bedrooms}
                    </span>
                    <span>
                      <Bath className="w-4 h-4" /> {property.bathrooms}
                    </span>
                    <span>
                      <Maximize className="w-4 h-4" /> {property.area} sq ft
                    </span>
                  </div>
                  <div className="property-footer">
                    <span className="property-price">
                      ${property.price.toLocaleString()}
                    </span>

                    <div className="flex gap-2">
                      <Button
                        onClick={() => navigate(`/property/${property.id}`)}
                      >
                        View
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => handleRemove(property.id)}
                        data-testid="remove-favorite-btn"
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------- PROPERTY DETAIL PAGE (with Favorites + Reviews) ----------------

function PropertyPage() {
  const { id } = useParams();
  const { user, token } = useAuth();
  const navigate = useNavigate();

  const [property, setProperty] = useState(null);
  const [bookingData, setBookingData] = useState({
    check_in: "",
    check_out: "",
    guests: 1,
    message: "",
  });
  const [showBookingDialog, setShowBookingDialog] = useState(false);

  // Favorites
  const [isFavorite, setIsFavorite] = useState(false);
  const [favoriteId, setFavoriteId] = useState(null);

  // Reviews
  const [reviews, setReviews] = useState([]);
  const [avgRating, setAvgRating] = useState(null);
  const [showReviewDialog, setShowReviewDialog] = useState(false);
  const [reviewData, setReviewData] = useState({
    rating: 5,
    comment: "",
  });

  useEffect(() => {
    const fetchProperty = async () => {
      try {
        const response = await axios.get(`${API}/properties/${id}`);
        setProperty(response.data);
      } catch (error) {
        toast.error("Failed to load property");
      }
    };

    fetchProperty();
  }, [id]);

  // Fetch reviews for this property
  useEffect(() => {
    const fetchReviews = async () => {
      try {
        const res = await axios.get(`${API}/properties/${id}/reviews`);
        const list = res.data || [];
        setReviews(list);
        if (list.length > 0) {
          const sum = list.reduce((acc, r) => acc + (r.rating || 0), 0);
          setAvgRating(sum / list.length);
        } else {
          setAvgRating(null);
        }
      } catch (err) {
        console.error("Failed to load reviews", err);
      }
    };

    fetchReviews();
  }, [id]);

  // Fetch favorites for current user to see if this property is saved
  useEffect(() => {
    if (!user || !token) {
      setIsFavorite(false);
      setFavoriteId(null);
      return;
    }

    const fetchFavorites = async () => {
      try {
        const res = await axios.get(`${API}/favorites`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const list = res.data || [];
        const fav = list.find((f) => f.property_id === id);
        if (fav) {
          setIsFavorite(true);
          setFavoriteId(fav.id);
        } else {
          setIsFavorite(false);
          setFavoriteId(null);
        }
      } catch (err) {
        console.error("Failed to load favorites", err);
      }
    };

    fetchFavorites();
  }, [id, user, token]);

  const handleBooking = async (e) => {
    e.preventDefault();
    if (!user) {
      toast.error("Please login to book a property");
      navigate("/login");
      return;
    }

    try {
      await axios.post(
        `${API}/bookings`,
        {
          property_id: id,
          ...bookingData,
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      toast.success("Booking request submitted successfully!");
      setShowBookingDialog(false);
      navigate("/bookings");
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to create booking");
    }
  };

  const handleToggleFavorite = async () => {
    if (!user || !token) {
      toast.error("Please login to save properties");
      navigate("/login");
      return;
    }

    try {
      if (!isFavorite) {
        const res = await axios.post(
          `${API}/favorites`,
          { property_id: id },
          { headers: { Authorization: `Bearer ${token}` } }
        );
        setIsFavorite(true);
        setFavoriteId(res.data.id);
        toast.success("Property added to favorites");
      } else {
        if (!favoriteId) return;
        await axios.delete(`${API}/favorites/${favoriteId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setIsFavorite(false);
        setFavoriteId(null);
        toast.success("Property removed from favorites");
      }
    } catch (err) {
      toast.error("Failed to update favorites");
    }
  };

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    if (!user || !token) {
      toast.error("Please login to write a review");
      navigate("/login");
      return;
    }

    try {
      await axios.post(
        `${API}/reviews`,
        {
          property_id: id,
          rating: reviewData.rating,
          comment: reviewData.comment,
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      toast.success("Review submitted");
      setShowReviewDialog(false);
      setReviewData({ rating: 5, comment: "" });

      const res = await axios.get(`${API}/properties/${id}/reviews`);
      const list = res.data || [];
      setReviews(list);
      if (list.length > 0) {
        const sum = list.reduce((acc, r) => acc + (r.rating || 0), 0);
        setAvgRating(sum / list.length);
      } else {
        setAvgRating(null);
      }
    } catch (err) {
      toast.error("Failed to submit review");
    }
  };

  if (!property) return <div className="loading-screen">Loading...</div>;

  return (
    <div className="page-container">
      <Navigation />

      <div className="property-detail" data-testid="property-detail">
        <div className="property-detail-image">
          <img src={property.image_url} alt={property.title} />
        </div>

        <div className="property-detail-content">
          <div className="property-detail-header">
            <div>
              <h1 data-testid="property-detail-title">{property.title}</h1>
              <div className="property-location">
                <MapPin className="w-5 h-5" />
                <span>{property.address}</span>
              </div>

              {avgRating !== null && (
                <div className="property-rating">
                  <Star className="w-5 h-5 text-yellow-500" />
                  <span>
                    {avgRating.toFixed(1)} / 5 • {reviews.length} review
                    {reviews.length !== 1 ? "s" : ""}
                  </span>
                </div>
              )}
            </div>

            <div
              className="property-detail-price"
              data-testid="property-detail-price"
            >
              ${property.price.toLocaleString()}
            </div>
          </div>

          <div className="property-detail-features">
            <div className="feature-item">
              <Bed className="w-6 h-6" />
              <span>{property.bedrooms} Bedrooms</span>
            </div>
            <div className="feature-item">
              <Bath className="w-6 h-6" />
              <span>{property.bathrooms} Bathrooms</span>
            </div>
            <div className="feature-item">
              <Maximize className="w-6 h-6" />
              <span>{property.area} sq ft</span>
            </div>
            <div className="feature-item">
              <Building2 className="w-6 h-6" />
              <span>{property.property_type || "Property"}</span>
            </div>
          </div>

          <div className="property-detail-section">
            <h2>Description</h2>
            <p>{property.description}</p>
          </div>

          <div className="property-detail-section">
            <h2>Amenities</h2>
            <div className="amenities-list">
              {property.amenities.map((amenity, idx) => (
                <span key={idx} className="amenity-badge">
                  {amenity}
                </span>
              ))}
            </div>
          </div>

          <div className="property-detail-actions">
            <Button
              size="lg"
              className="booking-btn"
              onClick={() => setShowBookingDialog(true)}
              data-testid="book-now-btn"
            >
              Book Now
            </Button>

            <Button
              variant={isFavorite ? "default" : "outline"}
              size="lg"
              className="favorite-btn ml-3"
              onClick={handleToggleFavorite}
              data-testid="favorite-btn"
            >
              <Heart
                className="w-5 h-5 mr-2"
                fill={isFavorite ? "currentColor" : "none"}
              />
              {isFavorite ? "Saved" : "Save"}
            </Button>
          </div>

          <div className="property-detail-section">
            <div className="reviews-header">
              <h2>Reviews</h2>
              {avgRating !== null && reviews.length > 0 && (
                <div className="reviews-summary">
                  <Star className="w-4 h-4 text-yellow-500" />
                  <span>
                    {avgRating.toFixed(1)} / 5 • {reviews.length} review
                    {reviews.length !== 1 ? "s" : ""}
                  </span>
                </div>
              )}
            </div>

            {reviews.length === 0 ? (
              <p>No reviews yet. Be the first to review this property.</p>
            ) : (
              <div className="reviews-list">
                {reviews.map((rev) => (
                  <div key={rev.id} className="review-card">
                    <div className="review-header">
                      <strong>{rev.user_name}</strong>
                      <div className="review-rating">
                        {Array.from({ length: rev.rating }).map((_, i) => (
                          <Star key={i} className="w-4 h-4 text-yellow-500" />
                        ))}
                      </div>
                    </div>
                    {rev.comment && <p className="review-comment">{rev.comment}</p>}
                    <span className="review-date">
                      {rev.created_at ? new Date(rev.created_at).toLocaleDateString() : ""}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <Button
              className="mt-4"
              data-testid="write-review-btn"
              onClick={() => {
                if (!user) {
                  toast.error("Please login to write a review");
                  navigate("/login");
                  return;
                }
                setShowReviewDialog(true);
              }}
            >
              Write a Review
            </Button>
          </div>
        </div>
      </div>

      {/* Booking Dialog */}
      <Dialog open={showBookingDialog} onOpenChange={setShowBookingDialog}>
        <DialogContent data-testid="booking-dialog">
          <DialogHeader>
            <DialogTitle>Book This Property</DialogTitle>
            <DialogDescription>
              Fill in the details to request a booking
            </DialogDescription>
          </DialogHeader>

          <form onSubmit={handleBooking} className="booking-form">
            <div className="form-group">
              <Label htmlFor="check_in">Check-in Date</Label>
              <Input
                id="check_in"
                type="date"
                required
                value={bookingData.check_in}
                onChange={(e) =>
                  setBookingData({ ...bookingData, check_in: e.target.value })
                }
                data-testid="check-in-input"
              />
            </div>

            <div className="form-group">
              <Label htmlFor="check_out">Check-out Date</Label>
              <Input
                id="check_out"
                type="date"
                required
                value={bookingData.check_out}
                onChange={(e) =>
                  setBookingData({ ...bookingData, check_out: e.target.value })
                }
                data-testid="check-out-input"
              />
            </div>

            <div className="form-group">
              <Label htmlFor="guests">Number of Guests</Label>
              <Input
                id="guests"
                type="number"
                min="1"
                required
                value={bookingData.guests}
                onChange={(e) =>
                  setBookingData({
                    ...bookingData,
                    guests: parseInt(e.target.value) || 1,
                  })
                }
                data-testid="guests-input"
              />
            </div>

            <div className="form-group">
              <Label htmlFor="message">Message (Optional)</Label>
              <Textarea
                id="message"
                placeholder="Any special requests or questions..."
                value={bookingData.message}
                onChange={(e) =>
                  setBookingData({ ...bookingData, message: e.target.value })
                }
                data-testid="message-input"
              />
            </div>

            <Button type="submit" className="w-full" data-testid="submit-booking-btn">
              Submit Booking
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      {/* Review Dialog */}
      <Dialog open={showReviewDialog} onOpenChange={setShowReviewDialog}>
        <DialogContent data-testid="review-dialog">
          <DialogHeader>
            <DialogTitle>Write a Review</DialogTitle>
            <DialogDescription>
              Share your experience about this property
            </DialogDescription>
          </DialogHeader>

          <form onSubmit={handleSubmitReview} className="booking-form">
            <div className="form-group">
              <Label htmlFor="rating">Rating (1–5)</Label>
              <Select
                value={String(reviewData.rating)}
                onValueChange={(val) =>
                  setReviewData({ ...reviewData, rating: parseInt(val) })
                }
              >
                <SelectTrigger id="rating">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="3">3</SelectItem>
                  <SelectItem value="4">4</SelectItem>
                  <SelectItem value="5">5</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="form-group">
              <Label htmlFor="review-comment">Comment</Label>
              <Textarea
                id="review-comment"
                placeholder="Write your feedback..."
                value={reviewData.comment}
                onChange={(e) =>
                  setReviewData({ ...reviewData, comment: e.target.value })
                }
              />
            </div>

            <Button type="submit" className="w-full" data-testid="submit-review-btn">
              Submit Review
            </Button>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ---------------- PROPERTIES LIST PAGE ----------------

function PropertiesPage() {
  const navigate = useNavigate();
  const [properties, setProperties] = useState([]);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    fetchProperties();
    // eslint-disable-next-line
  }, []);

  const fetchProperties = async () => {
    try {
      const response = await axios.get(`${API}/properties`);
      setProperties(response.data);
    } catch (error) {
      toast.error("Failed to load properties");
    }
  };

  const filteredProperties =
    filter === "all"
      ? properties
      : properties.filter(
          (p) => (p.property_type || "").toLowerCase() === filter.toLowerCase()
        );

  return (
    <div className="page-container">
      <Navigation />

      <div className="properties-page">
        <div className="page-header">
          <h1>All Properties</h1>
          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger className="w-48" data-testid="property-filter">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="villa">Villa</SelectItem>
              <SelectItem value="apartment">Apartment</SelectItem>
              <SelectItem value="house">House</SelectItem>
              <SelectItem value="penthouse">Penthouse</SelectItem>
              <SelectItem value="condo">Condo</SelectItem>
              <SelectItem value="cabin">Cabin</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="properties-grid">
          {filteredProperties.map((property) => (
            <Card key={property.id} className="property-card">
              <div
                className="property-image"
                onClick={() => navigate(`/property/${property.id}`)}
              >
                <img src={property.image_url} alt={property.title} />
                <div className="property-badge">{property.status}</div>
              </div>
              <CardContent className="property-info">
                <h3 className="property-title">{property.title}</h3>
                <div className="property-location">
                  <MapPin className="w-4 h-4" />
                  <span>{property.location}</span>
                </div>
                <div className="property-features">
                  <span>
                    <Bed className="w-4 h-4" /> {property.bedrooms}
                  </span>
                  <span>
                    <Bath className="w-4 h-4" /> {property.bathrooms}
                  </span>
                  <span>
                    <Maximize className="w-4 h-4" /> {property.area} sq ft
                  </span>
                </div>
                <div className="property-footer">
                  <span className="property-price">
                    ${property.price.toLocaleString()}
                  </span>
                  <Button onClick={() => navigate(`/property/${property.id}`)}>
                    View Details
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------- BOOKINGS PAGE ----------------

function BookingsPage() {
  const { token } = useAuth();
  const [bookings, setBookings] = useState([]);
  const [properties, setProperties] = useState({});

  useEffect(() => {
    fetchBookings();
    // eslint-disable-next-line
  }, []);

  const fetchBookings = async () => {
    try {
      const response = await axios.get(`${API}/bookings`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setBookings(response.data);

      const propertyPromises = response.data.map((b) =>
        axios.get(`${API}/properties/${b.property_id}`).catch(() => null)
      );
      const propertyResponses = await Promise.all(propertyPromises);
      const propsMap = {};
      propertyResponses.forEach((res) => {
        if (res?.data) propsMap[res.data.id] = res.data;
      });
      setProperties(propsMap);
    } catch (error) {
      toast.error("Failed to load bookings");
    }
  };

  const handleCancelBooking = async (bookingId) => {
    if (!window.confirm("Are you sure you want to cancel this booking?")) return;

    try {
      await axios.delete(`${API}/bookings/${bookingId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      toast.success("Booking cancelled successfully");
      fetchBookings();
    } catch (error) {
      toast.error("Failed to cancel booking");
    }
  };

  return (
    <div className="page-container">
      <Navigation />

      <div className="bookings-page">
        <h1>My Bookings</h1>

        {bookings.length === 0 ? (
          <Card className="empty-state" data-testid="empty-bookings">
            <CardContent>
              <Calendar className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>You haven't made any bookings yet</p>
              <Button
                onClick={() => (window.location.href = "/properties")}
                className="mt-4"
              >
                Browse Properties
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="bookings-list" data-testid="bookings-list">
            {bookings.map((booking) => {
              const property = properties[booking.property_id];
              return (
                <Card
                  key={booking.id}
                  className="booking-card"
                  data-testid={`booking-card-${booking.id}`}
                >
                  <CardContent className="booking-content">
                    {property && (
                      <div className="booking-image">
                        <img src={property.image_url} alt={property.title} />
                      </div>
                    )}
                    <div className="booking-details">
                      <h3>{property?.title || "Property"}</h3>
                      <div className="booking-info-grid">
                        <div>
                          <strong>Check-in:</strong> {booking.check_in}
                        </div>
                        <div>
                          <strong>Check-out:</strong> {booking.check_out}
                        </div>
                        <div>
                          <strong>Guests:</strong> {booking.guests}
                        </div>
                        <div>
                          <strong>Status:</strong>
                          <span
                            className={`status-badge status-${booking.status}`}
                            data-testid="booking-status"
                          >
                            {booking.status}
                          </span>
                        </div>
                      </div>
                      {booking.message && (
                        <div className="booking-message">
                          <strong>Message:</strong> {booking.message}
                        </div>
                      )}
                    </div>
                    <div className="booking-actions">
                      <Button
                        variant="destructive"
                        onClick={() => handleCancelBooking(booking.id)}
                        data-testid="cancel-booking-btn"
                      >
                        Cancel
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------- OWNER DASHBOARD ----------------

function OwnerDashboardPage() {
  const { user, token } = useAuth();
  const navigate = useNavigate();

  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    price: "",
    location: "",
    address: "",
    bedrooms: "",
    bathrooms: "",
    area: "",
    image_url: "",
    amenities: "",
  });

  useEffect(() => {
    if (!user) return;
    if (user.role !== "owner" && user.role !== "admin") {
      toast.error("Only owners can access the Owner Dashboard");
      navigate("/");
    } else {
      fetchMyProperties();
    }
    // eslint-disable-next-line
  }, [user]);

  const fetchMyProperties = async () => {
    try {
      const res = await axios.get(`${API}/owner/my-properties`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setProperties(res.data);
    } catch (err) {
      toast.error("Failed to load your properties");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateProperty = async (e) => {
    e.preventDefault();
    setSaving(true);

    try {
      const payload = {
        title: formData.title,
        description: formData.description,
        price: parseFloat(formData.price || "0"),
        location: formData.location,
        address: formData.address,
        bedrooms: parseInt(formData.bedrooms || "0"),
        bathrooms: parseInt(formData.bathrooms || "0"),
        area: parseFloat(formData.area || "0"),
        image_url: formData.image_url,
        amenities: formData.amenities
          .split(",")
          .map((a) => a.trim())
          .filter(Boolean),
      };

      await axios.post(`${API}/properties`, payload, {
        headers: { Authorization: `Bearer ${token}` },
      });

      toast.success("Property created successfully");
      setFormData({
        title: "",
        description: "",
        price: "",
        location: "",
        address: "",
        bedrooms: "",
        bathrooms: "",
        area: "",
        image_url: "",
        amenities: "",
      });
      fetchMyProperties();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to create property");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="page-container">
      <Navigation />

      <div className="owner-page">
        <h1>Owner Dashboard</h1>
        <p className="page-subtitle">Manage your listings and add new properties.</p>

        <Tabs defaultValue="my-properties">
          <TabsList className="admin-tabs">
            <TabsTrigger value="my-properties">My Properties</TabsTrigger>
            <TabsTrigger value="add-property" data-owner-add-tab>
              Add New Property
            </TabsTrigger>
          </TabsList>

          <TabsContent value="my-properties">
            {loading ? (
              <div className="loading-screen">Loading your properties...</div>
            ) : properties.length === 0 ? (
              <Card className="empty-state">
                <CardContent>
                  <Building2 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p>You haven&apos;t added any properties yet.</p>
                  <Button
                    className="mt-4"
                    onClick={() =>
                      document.querySelector("[data-owner-add-tab]")?.click()
                    }
                  >
                    Add Your First Property
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="properties-grid">
                {properties.map((property) => (
                  <Card key={property.id} className="property-card">
                    <div className="property-image">
                      <img src={property.image_url} alt={property.title} />
                      <div className="property-badge">{property.status}</div>
                    </div>
                    <CardContent className="property-info">
                      <h3 className="property-title">{property.title}</h3>
                      <div className="property-location">
                        <MapPin className="w-4 h-4" />
                        <span>{property.location}</span>
                      </div>
                      <div className="property-features">
                        <span>
                          <Bed className="w-4 h-4" /> {property.bedrooms}
                        </span>
                        <span>
                          <Bath className="w-4 h-4" /> {property.bathrooms}
                        </span>
                        <span>
                          <Maximize className="w-4 h-4" /> {property.area} sq ft
                        </span>
                      </div>
                      <div className="property-footer">
                        <span className="property-price">
                          ${property.price.toLocaleString()}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="add-property">
            <Card>
              <CardHeader>
                <CardTitle>Add New Property</CardTitle>
                <CardDescription>
                  Fill in the details to create a new listing.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleCreateProperty} className="auth-form">
                  <div className="form-group">
                    <Label htmlFor="title">Title</Label>
                    <Input
                      id="title"
                      required
                      value={formData.title}
                      onChange={(e) =>
                        setFormData({ ...formData, title: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-group">
                    <Label htmlFor="location">Location</Label>
                    <Input
                      id="location"
                      required
                      value={formData.location}
                      onChange={(e) =>
                        setFormData({ ...formData, location: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-group">
                    <Label htmlFor="address">Address</Label>
                    <Input
                      id="address"
                      required
                      value={formData.address}
                      onChange={(e) =>
                        setFormData({ ...formData, address: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-group">
                    <Label htmlFor="price">Price</Label>
                    <Input
                      id="price"
                      type="number"
                      min="0"
                      required
                      value={formData.price}
                      onChange={(e) =>
                        setFormData({ ...formData, price: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <Label htmlFor="bedrooms">Bedrooms</Label>
                      <Input
                        id="bedrooms"
                        type="number"
                        min="0"
                        required
                        value={formData.bedrooms}
                        onChange={(e) =>
                          setFormData({ ...formData, bedrooms: e.target.value })
                        }
                      />
                    </div>

                    <div className="form-group">
                      <Label htmlFor="bathrooms">Bathrooms</Label>
                      <Input
                        id="bathrooms"
                        type="number"
                        min="0"
                        required
                        value={formData.bathrooms}
                        onChange={(e) =>
                          setFormData({ ...formData, bathrooms: e.target.value })
                        }
                      />
                    </div>

                    <div className="form-group">
                      <Label htmlFor="area">Area (sq ft)</Label>
                      <Input
                        id="area"
                        type="number"
                        min="0"
                        required
                        value={formData.area}
                        onChange={(e) =>
                          setFormData({ ...formData, area: e.target.value })
                        }
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <Label htmlFor="image_url">Image URL</Label>
                    <Input
                      id="image_url"
                      type="url"
                      required
                      value={formData.image_url}
                      onChange={(e) =>
                        setFormData({ ...formData, image_url: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-group">
                    <Label htmlFor="amenities">
                      Amenities (comma separated)
                    </Label>
                    <Input
                      id="amenities"
                      placeholder="Pool, Garden, Parking"
                      value={formData.amenities}
                      onChange={(e) =>
                        setFormData({ ...formData, amenities: e.target.value })
                      }
                    />
                  </div>

                  <div className="form-group">
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      required
                      value={formData.description}
                      onChange={(e) =>
                        setFormData({ ...formData, description: e.target.value })
                      }
                    />
                  </div>

                  <Button type="submit" className="w-full" disabled={saving}>
                    {saving ? "Saving..." : "Create Property"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// ---------------- LOGIN PAGE ----------------

function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`${API}/auth/login`, formData);
      const { access_token, user } = response.data;

      login(access_token, user);
      toast.success("Welcome back!");

      let redirect = "/";
      if (user.role === "owner") redirect = "/owner";
      if (user.role === "admin") redirect = "/admin";

      navigate(redirect);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page" data-testid="login-page">
      <div className="auth-container">
        <Card className="auth-card">
          <CardHeader>
            <div className="auth-logo">
              <Building2 className="w-12 h-12" />
            </div>
            <CardTitle>Welcome Back</CardTitle>
            <CardDescription>Login to access your account</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                  data-testid="email-input"
                />
              </div>
              <div className="form-group">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  required
                  value={formData.password}
                  onChange={(e) =>
                    setFormData({ ...formData, password: e.target.value })
                  }
                  data-testid="password-input"
                />
              </div>
              <Button
                type="submit"
                className="w-full"
                disabled={loading}
                data-testid="login-submit-btn"
              >
                {loading ? "Logging in..." : "Login"}
              </Button>
            </form>
            <div className="auth-footer">
              <p>
                Don't have an account? <a href="/register">Sign up</a>
              </p>
              <p className="demo-credentials">
                <small>Demo: admin@realestate.com / admin123</small>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------- REGISTER PAGE ----------------

function RegisterPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "",
    phone: "",
    role: "buyer",
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`${API}/auth/register`, formData);
      const { access_token, user } = response.data;

      login(access_token, user);
      toast.success("Account created successfully!");

      let redirect = "/";
      if (user.role === "owner") redirect = "/owner";
      if (user.role === "admin") redirect = "/admin";

      navigate(redirect);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page" data-testid="register-page">
      <div className="auth-container">
        <Card className="auth-card">
          <CardHeader>
            <div className="auth-logo">
              <Building2 className="w-12 h-12" />
            </div>
            <CardTitle>Create Account</CardTitle>
            <CardDescription>
              Sign up as a buyer or property owner (dealer)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  data-testid="name-input"
                />
              </div>

              <div className="form-group">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                  data-testid="email-input"
                />
              </div>

              <div className="form-group">
                <Label htmlFor="phone">Phone (Optional)</Label>
                <Input
                  id="phone"
                  type="tel"
                  value={formData.phone}
                  onChange={(e) =>
                    setFormData({ ...formData, phone: e.target.value })
                  }
                  data-testid="phone-input"
                />
              </div>

              <div className="form-group">
                <Label htmlFor="role">Account Type</Label>
                <Select
                  value={formData.role}
                  onValueChange={(value) =>
                    setFormData({ ...formData, role: value })
                  }
                >
                  <SelectTrigger id="role" data-testid="role-select">
                    <SelectValue placeholder="Select account type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="buyer">Buyer</SelectItem>
                    <SelectItem value="owner">Owner / Dealer</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="form-group">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  required
                  minLength="6"
                  value={formData.password}
                  onChange={(e) =>
                    setFormData({ ...formData, password: e.target.value })
                  }
                  data-testid="password-input"
                />
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={loading}
                data-testid="register-submit-btn"
              >
                {loading ? "Creating account..." : "Sign Up"}
              </Button>
            </form>
            <div className="auth-footer">
              <p>
                Already have an account? <a href="/login">Login</a>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------- ADMIN DASHBOARD ----------------

function AdminPage() {
  const { token } = useAuth();
  const [stats, setStats] = useState({});
  const [users, setUsers] = useState([]);
  const [properties, setProperties] = useState([]);
  const [bookings, setBookings] = useState([]);
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    seedData();
    fetchStats();
    fetchUsers();
    fetchProperties();
    fetchBookings();
    // eslint-disable-next-line
  }, []);

  const seedData = async () => {
    try {
      await axios.post(`${API}/admin/seed`);
    } catch (error) {
      console.log("Seed data already exists");
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/admin/stats`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setStats(response.data);
    } catch (error) {
      toast.error("Failed to load stats");
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await axios.get(`${API}/admin/users`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUsers(response.data);
    } catch (error) {
      toast.error("Failed to load users");
    }
  };

  const fetchProperties = async () => {
    try {
      const response = await axios.get(`${API}/properties`);
      setProperties(response.data);
    } catch (error) {
      toast.error("Failed to load properties");
    }
  };

  const fetchBookings = async () => {
    try {
      const response = await axios.get(`${API}/bookings`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setBookings(response.data);
    } catch (error) {
      toast.error("Failed to load bookings");
    }
  };

  const updateBookingStatus = async (bookingId, newStatus) => {
    try {
      await axios.put(
        `${API}/bookings/${bookingId}/status?status=${newStatus}`,
        {},
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      toast.success("Booking status updated");
      fetchBookings();
    } catch (error) {
      toast.error("Failed to update booking status");
    }
  };

  return (
    <div className="page-container">
      <Navigation />

      <div className="admin-page" data-testid="admin-page">
        <h1>Admin Dashboard</h1>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="admin-tabs">
            <TabsTrigger value="overview" data-testid="tab-overview">
              Overview
            </TabsTrigger>
            <TabsTrigger value="users" data-testid="tab-users">
              Users
            </TabsTrigger>
            <TabsTrigger value="properties" data-testid="tab-properties">
              Properties
            </TabsTrigger>
            <TabsTrigger value="bookings" data-testid="tab-bookings">
              Bookings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <div className="stats-grid" data-testid="stats-grid">
              <Card>
                <CardHeader>
                  <CardTitle>Total Users</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="stat-value" data-testid="stat-users">
                    {stats.total_users || 0}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Total Properties</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="stat-value" data-testid="stat-properties">
                    {stats.total_properties || 0}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Total Bookings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="stat-value" data-testid="stat-bookings">
                    {stats.total_bookings || 0}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Pending Bookings</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="stat-value" data-testid="stat-pending">
                    {stats.pending_bookings || 0}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="users">
            <Card>
              <CardHeader>
                <CardTitle>All Users</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="table-container" data-testid="users-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Email</th>
                        <th>Phone</th>
                        <th>Role</th>
                        <th>Joined</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.map((user) => (
                        <tr key={user.id}>
                          <td>{user.name}</td>
                          <td>{user.email}</td>
                          <td>{user.phone || "N/A"}</td>
                          <td>
                            <span className="role-badge">{user.role}</span>
                          </td>
                          <td>{new Date(user.created_at).toLocaleDateString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="properties">
            <Card>
              <CardHeader>
                <CardTitle>All Properties</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="table-container" data-testid="properties-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Title</th>
                        <th>Location</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Status</th>
                        <th>Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {properties.map((property) => (
                        <tr key={property.id}>
                          <td>{property.title}</td>
                          <td>{property.location}</td>
                          <td>{property.property_type || "-"}</td>
                          <td>${property.price.toLocaleString()}</td>
                          <td>
                            <span
                              className={`status-badge status-${property.status}`}
                            >
                              {property.status}
                            </span>
                          </td>
                          <td>
                            {new Date(property.created_at).toLocaleDateString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="bookings">
            <Card>
              <CardHeader>
                <CardTitle>All Bookings</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="table-container" data-testid="bookings-table">
                  <table>
                    <thead>
                      <tr>
                        <th>User</th>
                        <th>Property ID</th>
                        <th>Check-in</th>
                        <th>Check-out</th>
                        <th>Guests</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bookings.map((booking) => (
                        <tr key={booking.id}>
                          <td>{booking.user_name}</td>
                          <td>{booking.property_id.substring(0, 8)}...</td>
                          <td>{booking.check_in}</td>
                          <td>{booking.check_out}</td>
                          <td>{booking.guests}</td>
                          <td>
                            <span
                              className={`status-badge status-${booking.status}`}
                            >
                              {booking.status}
                            </span>
                          </td>
                          <td>
                            <Select
                              value={booking.status}
                              onValueChange={(val) =>
                                updateBookingStatus(booking.id, val)
                              }
                            >
                              <SelectTrigger className="w-32">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="pending">Pending</SelectItem>
                                <SelectItem value="confirmed">
                                  Confirmed
                                </SelectItem>
                                <SelectItem value="cancelled">
                                  Cancelled
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// ---------------- ROOT APP ----------------

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/properties" element={<PropertiesPage />} />
          <Route path="/property/:id" element={<PropertyPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* ✅ NEW: Favorites route */}
          <Route
            path="/favorites"
            element={
              <ProtectedRoute>
                <FavoritesPage />
              </ProtectedRoute>
            }
          />

          <Route
            path="/bookings"
            element={
              <ProtectedRoute>
                <BookingsPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/owner"
            element={
              <ProtectedRoute>
                <OwnerDashboardPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin"
            element={
              <ProtectedRoute adminOnly>
                <AdminPage />
              </ProtectedRoute>
            }
          />
        </Routes>

        <Toaster position="top-right" richColors />
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
