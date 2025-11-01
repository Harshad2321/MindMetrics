/**
 * Digital Twin of Stress - Frontend JavaScript
 * Handles form submission, API communication, and result display
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Theme Management
let isDarkMode = false;
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.querySelector('.theme-icon');
const modelInfoBadge = document.getElementById('modelInfoBadge');

// DOM Elements - Form
const stressForm = document.getElementById('stressForm');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorText');
const closeAlertBtn = document.getElementById('closeAlert');
const predictBtn = document.getElementById('predictBtn');
const sampleBtn = document.getElementById('sampleBtn');
const resetBtn = document.getElementById('resetBtn');

// Result display elements
const resultCard = document.getElementById('resultCard');
const stressEmoji = document.getElementById('stressEmoji');
const stressScore = document.getElementById('stressScore');
const categoryBadge = document.getElementById('categoryBadge');
const confidenceBadge = document.getElementById('confidenceBadge');
const progressFill = document.getElementById('progressFill');
const suggestionText = document.getElementById('suggestionText');

/**
 * Show/hide UI elements
 */
function showElement(element) {
    if (element) {
        element.classList.remove('hidden');
        element.style.display = '';
    }
}

function hideElement(element) {
    if (element) {
        element.classList.add('hidden');
        element.style.display = 'none';
    }
}

/**
 * Display error message to user
 */
function showError(message) {
    errorMessage.textContent = message;
    showElement(errorAlert);
}

/**
 * Close error alert
 */
if (closeAlertBtn) {
    closeAlertBtn.addEventListener('click', () => {
        hideElement(errorAlert);
    });
}

/**
 * Validate form inputs
 */
function validateInputs(data) {
    // Age validation (1-120 years)
    if (data.age < 1 || data.age > 120) {
        throw new Error('Age must be between 1 and 120 years');
    }

    // Heart rate validation (40-200 BPM)
    if (data.heart_rate < 40 || data.heart_rate > 200) {
        throw new Error('Heart rate must be between 40 and 200 BPM');
    }

    // Sleep duration validation (0-24 hours)
    if (data.sleep_duration < 0 || data.sleep_duration > 24) {
        throw new Error('Sleep duration must be between 0 and 24 hours');
    }

    // Step count validation (0-50000 steps)
    if (data.step_count < 0 || data.step_count > 50000) {
        throw new Error('Step count must be between 0 and 50,000');
    }

    return true;
}

/**
 * Collect form data and prepare for API request
 */
function getFormData() {
    const formData = {
        age: parseInt(document.getElementById('age').value),
        gender: document.getElementById('gender').value,
        heart_rate: parseFloat(document.getElementById('heartRate').value),
        sleep_duration: parseFloat(document.getElementById('sleepDuration').value),
        step_count: parseInt(document.getElementById('stepCount').value),
        sleep_quality: document.getElementById('sleepQuality').value,
        activity_level: document.getElementById('activityLevel').value
    };

    return formData;
}

/**
 * Send prediction request to backend API
 */
async function predictStress(data) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }

    return await response.json();
}

/**
 * Display prediction results with animation and color coding
 */
function displayResults(result) {
    // Update stress score
    stressScore.textContent = result.predicted_stress.toFixed(1);
    
    // Update category badge
    categoryBadge.textContent = result.stress_category;
    
    // Update confidence badge
    if (result.confidence !== null && result.confidence !== undefined) {
        confidenceBadge.textContent = `Confidence: ${result.confidence}%`;
        showElement(confidenceBadge);
    } else {
        hideElement(confidenceBadge);
    }
    
    // Update suggestion
    suggestionText.textContent = result.suggestion;
    
    // Update progress bar (0-10 scale)
    const percentage = (result.predicted_stress / 10) * 100;
    progressFill.style.width = `${percentage}%`;

    // Apply color coding to result card (RED / ORANGE / GREEN)
    resultCard.classList.remove('stress-high', 'stress-moderate', 'stress-low');
    
    if (result.predicted_stress >= 7) {
        // HIGH STRESS - RED
        resultCard.classList.add('stress-high');
    } else if (result.predicted_stress >= 4) {
        // MODERATE STRESS - ORANGE
        resultCard.classList.add('stress-moderate');
    } else {
        // LOW STRESS - GREEN
        resultCard.classList.add('stress-low');
    }

    // Show results section
    showElement(resultsSection);

    // Scroll to results smoothly
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

/**
 * Handle form submission
 */
stressForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    try {
        // Hide any previous errors
        hideElement(errorAlert);

        // Get form data
        const formData = getFormData();

        // Validate inputs
        validateInputs(formData);

        // Show loading overlay
        showElement(loadingOverlay);
        predictBtn.disabled = true;

        // Make API request
        const result = await predictStress(formData);

        // Hide loading overlay
        hideElement(loadingOverlay);

        // Display results
        displayResults(result);

    } catch (error) {
        console.error('Prediction error:', error);
        hideElement(loadingOverlay);
        predictBtn.disabled = false;

        // Check if it's a network error
        if (error.message.includes('fetch') || error.message.includes('Failed to fetch')) {
            showError('Cannot connect to server. Make sure the backend is running on http://localhost:8000');
        } else {
            showError(error.message);
        }
    }
});

/**
 * Handle reset button - show form again
 */
resetBtn.addEventListener('click', () => {
    // Hide results
    hideElement(resultsSection);

    // Re-enable predict button
    predictBtn.disabled = false;

    // Clear form
    stressForm.reset();

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

/**
 * Fetch and display model information
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (response.ok) {
            const modelInfo = await response.json();
            const badgeText = modelInfoBadge.querySelector('.badge-text');
            badgeText.textContent = `${modelInfo.model_name} | ${modelInfo.prediction_method} | Accuracy: ${modelInfo.accuracy}`;
            console.log('✅ Model Info:', modelInfo);
        }
    } catch (error) {
        console.warn('⚠️ Could not load model info');
        const badgeText = modelInfoBadge.querySelector('.badge-text');
        badgeText.textContent = 'Model info unavailable';
    }
}

/**
 * Check backend health on page load
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend is running and healthy');
            loadModelInfo(); // Load model info after confirming backend is up
        }
    } catch (error) {
        console.warn('⚠️ Backend is not accessible. Make sure to start the FastAPI server.');
    }
}

/**
 * Toggle dark/light mode
 */
function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode');
    
    // Update icon - switch between moon and sun SVG
    if (isDarkMode) {
        themeIcon.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>`;
    } else {
        themeIcon.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            </svg>`;
    }
    
    // Save preference
    localStorage.setItem('darkMode', isDarkMode);
    
    console.log(`Theme switched to ${isDarkMode ? 'dark' : 'light'} mode`);
}

/**
 * Load saved theme preference
 */
function loadThemePreference() {
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme === 'true') {
        isDarkMode = true;
        document.body.classList.add('dark-mode');
        themeIcon.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>`;
    }
}

// Theme toggle event listener
if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
}

// Initialize: Check backend health when page loads
window.addEventListener('load', () => {
    loadThemePreference();
    checkBackendHealth();
    console.log('🧠 Digital Twin of Stress - Frontend Loaded');
});

/**
 * Fill sample data for quick testing during viva
 */
function fillSampleData() {
    document.getElementById('age').value = 25;
    document.getElementById('gender').value = 'Male';
    document.getElementById('heartRate').value = 78;
    document.getElementById('sleepDuration').value = 6.5;
    document.getElementById('stepCount').value = 5000;
    document.getElementById('sleepQuality').value = 'Average';
    document.getElementById('activityLevel').value = 'Active';
    console.log('✅ Sample data filled!');
}

/**
 * Sample button click handler
 */
if (sampleBtn) {
    sampleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fillSampleData();
    });
}

// Press 'S' key to fill sample data (for quick demo)
document.addEventListener('keydown', (e) => {
    if ((e.key === 's' || e.key === 'S') && e.target.tagName !== 'INPUT') {
        fillSampleData();
    }
});
