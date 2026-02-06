const API_URL = "http://localhost:8000/api/v1";

// DOM Elements
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const authCard = document.getElementById('authCard');
const loginBtn = document.getElementById('loginBtn');
const signupBtn = document.getElementById('signupBtn');

// Toggle between Login and Signup
function toggleAuthMode() {
    if (loginForm.classList.contains('hidden')) {
        // Switch to Login
        signupForm.classList.add('hidden');
        setTimeout(() => {
            signupForm.style.display = 'none';
            loginForm.style.display = 'block';
            setTimeout(() => loginForm.classList.remove('hidden'), 50);
        }, 300);
    } else {
        // Switch to Signup
        loginForm.classList.add('hidden');
        setTimeout(() => {
            loginForm.style.display = 'none';
            signupForm.style.display = 'block';
            setTimeout(() => signupForm.classList.remove('hidden'), 50);
        }, 300);
    }
}

// Password Visibility Toggle
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    const icon = input.nextElementSibling;
    
    if (input.type === "password") {
        input.type = "text";
        icon.textContent = "visibility_off";
    } else {
        input.type = "password";
        icon.textContent = "visibility";
    }
}

// Show Toast Notification
function showToast(message, type = 'info') {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.style.backgroundColor = type === 'error' ? '#ef4444' : type === 'success' ? '#22c55e' : '#333';
    toast.className = "toast show";
    setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
}

// Login Handler
document.getElementById('loginFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(loginBtn, true);
    
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    const rememberMe = document.getElementById('rememberMe').checked;

    try {
        const response = await fetch(`${API_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (data.success) {
            showToast('Login Successful!', 'success');
            // Store token
            if (data.user && data.user.access_token) {
                if (rememberMe) {
                    localStorage.setItem('token', data.user.access_token);
                    localStorage.setItem('user', JSON.stringify(data.user));
                } else {
                    sessionStorage.setItem('token', data.user.access_token);
                    sessionStorage.setItem('user', JSON.stringify(data.user));
                }
            }
            // Redirect or update UI
            setTimeout(() => {
                window.location.href = 'dashboard.html'; // Or wherever you want to go
            }, 1000);
        } else {
            showToast(data.message || 'Login failed', 'error');
        }
    } catch (error) {
        showToast('Connection error. Is the server running?', 'error');
        console.error(error);
    } finally {
        setLoading(loginBtn, false);
    }
});

// Signup Handler
document.getElementById('signupFormElement').addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(signupBtn, true);
    
    const username = document.getElementById('signupUsername').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const age = document.getElementById('signupAge').value;
    const gender = document.getElementById('signupGender').value;

    const payload = {
        username,
        email,
        password,
        age: parseInt(age),
        height: 175, // Default for now
        weight: 70, // Default for now
        gender: gender,
        goal: "Fitness" // Default
    };

    try {
        const response = await fetch(`${API_URL}/auth/signup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            showToast('Account created! Logging you in...', 'success');
             // Store token (auto login)
             if (data.user && data.user.access_token) {
                localStorage.setItem('token', data.user.access_token);
                localStorage.setItem('user', JSON.stringify(data.user));
            }
            setTimeout(() => {
                window.location.href = 'dashboard.html';
            }, 1500);
        } else {
            showToast(data.message || 'Signup failed', 'error');
        }
    } catch (error) {
        showToast('Connection error. Is the server running?', 'error');
        console.error(error);
    } finally {
        setLoading(signupBtn, false);
    }
});

// Forgot Password Handler
async function showForgotPassword() {
    const email = prompt("Enter your email address to reset password:");
    if (email) {
        try {
            const formData = new FormData();
            formData.append('email', email);
            
            const response = await fetch(`${API_URL}/auth/forgot-password`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            showToast(data.message, data.success ? 'success' : 'error');
        } catch (error) {
            showToast('Failed to send request', 'error');
        }
    }
}

// Utility: Toggle Loading State
function setLoading(btn, isLoading) {
    if (isLoading) {
        btn.classList.add('loading');
        btn.disabled = true;
    } else {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}
