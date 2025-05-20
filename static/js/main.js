/**
 * SkyWings Flight Recommendation System
 * Main JavaScript file for handling API calls and frontend interactions
 */

// API Utility Functions
const API = {
    // Get recommendations from the backend
    getRecommendations: async function(searchParams) {
        try {
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(searchParams)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            throw error;
        }
    },
    
    // Update user profile
    updateProfile: async function(profileData) {
        try {
            const response = await fetch('/update_profile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(profileData)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error updating profile:', error);
            throw error;
        }
    },
    
    // Get user profile data
    getProfile: async function() {
        try {
            const response = await fetch('/get_profile');
            return await response.json();
        } catch (error) {
            console.error('Error fetching profile data:', error);
            throw error;
        }
    },
    
    // Perform login
    login: async function(credentials) {
        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(credentials)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error during login:', error);
            throw error;
        }
    },
    
    // Perform logout
    logout: async function() {
        try {
            const response = await fetch('/logout');
            return await response.json();
        } catch (error) {
            console.error('Error during logout:', error);
            throw error;
        }
    }
};

// UI Utility Functions
const UI = {
    // Show loading indicator
    showLoading: function() {
        document.getElementById('loadingState').classList.remove('hidden');
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('recommendationsSection').classList.add('hidden');
    },
    
    // Hide loading indicator
    hideLoading: function() {
        document.getElementById('loadingState').classList.add('hidden');
    },
    
    // Show recommendations
    showRecommendations: function(recommendations) {
        const container = document.getElementById('recommendationsContainer');
        
        // Clear previous recommendations
        container.innerHTML = '';
        
        if (recommendations && recommendations.length > 0) {
            document.getElementById('recommendationsSection').classList.remove('hidden');
            
            // Create and append recommendation cards
            recommendations.forEach((rec, index) => {
                const card = this.createRecommendationCard(rec, index);
                container.appendChild(card);
            });
            
            // Reset filters to show all
            document.querySelector('.filter-btn[data-class="Any"]').click();
        } else {
            document.getElementById('emptyState').classList.remove('hidden');
        }
    },
    
    // Create a recommendation card
    createRecommendationCard: function(recommendation, index) {
        // Clone the template
        const template = document.getElementById('recommendationCardTemplate');
        const card = document.importNode(template.content, true).querySelector('.flight-card');
        
        // Apply data attributes for filtering
        card.setAttribute('data-class', recommendation.class);
        
        // Set card content
        this.setCardContent(card, recommendation, index);
        
        return card;
    },
    
    // Set content for a recommendation card
    setCardContent: function(card, recommendation, index) {
        // Route-based images
        const routeImages = {
            'Asia': 'https://images.unsplash.com/photo-1542051841857-5f90071e7989',
            'Europe': 'https://images.unsplash.com/photo-1499856871958-5b9627545d1a',
            'America': 'https://images.unsplash.com/photo-1533929736458-ca588d08c8be',
            'North America': 'https://images.unsplash.com/photo-1533929736458-ca588d08c8be',
            'South America': 'https://images.unsplash.com/photo-1536196325145-32a73cd57e20',
            'Oceania': 'https://images.unsplash.com/photo-1518391846015-55a9cc003b25',
            'Africa': 'https://images.unsplash.com/photo-1547471080-91cb0f5300e0'
        };
        
        // Get image based on route
        let routeKey = 'Asia'; // Default
        if (recommendation.route) {
            const routeParts = recommendation.route.split(' to ');
            if (routeParts.length > 0 && routeImages[routeParts[0]]) {
                routeKey = routeParts[0];
            }
        }
        
        // Set image
        const img = card.querySelector('.card-img');
        img.src = `${routeImages[routeKey]}?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80`;
        img.alt = recommendation.route || 'Flight route';
        
        // Set badges for special recommendations
        const badge = card.querySelector('.card-badge');
        if (index === 0) {
            badge.textContent = 'Best Value';
            badge.classList.add('bg-blue-600', 'text-white');
            badge.classList.remove('hidden');
        } else if (recommendation.value_rating && parseFloat(recommendation.value_rating) >= 4.5) {
            badge.textContent = 'Best Rated';
            badge.classList.add('bg-green-500', 'text-white');
            badge.classList.remove('hidden');
        }
        
        // Set main content
        card.querySelector('.card-title').textContent = `${recommendation.airline}`;
        card.querySelector('.card-details').textContent = `${recommendation.class} • ${recommendation.season || 'Year-round'}`;
        card.querySelector('.card-price').textContent = recommendation.price;
        
        // Add discount for first recommendation
        if (index === 0) {
            const originalPrice = parseInt(recommendation.price.replace('$', '')) * 1.2;
            card.querySelector('.card-original-price').textContent = `$${Math.round(originalPrice)}`;
            card.querySelector('.card-original-price').classList.remove('hidden');
        }
        
        // Set route information
        card.querySelector('.card-route').textContent = recommendation.route;
        
        // Set date (current date + random days)
        const randomDays = Math.floor(Math.random() * 30) + 1;
        const futureDate = new Date();
        futureDate.setDate(futureDate.getDate() + randomDays);
        const formattedDate = futureDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        card.querySelector('.card-date').textContent = formattedDate;
        
        // Set ratings
        card.querySelector('.card-rating').textContent = recommendation.rating;
        if (recommendation.value_rating) {
            card.querySelector('.card-value-rating').textContent = `Value: ${recommendation.value_rating}`;
        }
        
        // Set airline
        card.querySelector('.card-airline').textContent = recommendation.airline;
    },
    
    // Show error message
    showError: function(message) {
        alert(message || 'An error occurred. Please try again.');
    }
};

// Event handlers for common UI elements
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers to filter buttons
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active button styles
            filterButtons.forEach(btn => {
                btn.classList.remove('bg-blue-100', 'text-blue-600');
                btn.classList.add('bg-gray-100', 'text-gray-600');
            });
            this.classList.remove('bg-gray-100', 'text-gray-600');
            this.classList.add('bg-blue-100', 'text-blue-600');
            
            // Filter visible cards
            const selectedClass = this.getAttribute('data-class');
            const cards = document.querySelectorAll('.flight-card');
            
            cards.forEach(card => {
                const cardClass = card.getAttribute('data-class');
                if (selectedClass === 'Any' || cardClass === selectedClass) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
        });
    });
    
    // Add click handlers to popular destinations
    const popularDestinations = document.querySelectorAll('.group.cursor-pointer');
    popularDestinations.forEach(destination => {
        destination.addEventListener('click', function() {
            const from = this.getAttribute('data-from');
            const to = this.getAttribute('data-to');
            
            // Set search form values
            document.getElementById('originSelect').value = from;
            document.getElementById('destinationSelect').value = to;
            
            // Scroll to search button
            document.getElementById('searchBtn').scrollIntoView({ behavior: 'smooth' });
        });
    });
    
    // Initialize today's date in departure date field
    const today = new Date();
    const departureDate = document.getElementById('departureDate');
    if (departureDate) {
        departureDate.min = today.toISOString().split('T')[0];
        departureDate.value = today.toISOString().split('T')[0];
    }
});