// Language translations
const translations = {
    en: {
        // Navigation
        'nav.home': 'Home',
        'nav.features': 'Features',
        'nav.services': 'Services',
        'nav.testimonials': 'Testimonials',
        'nav.contact': 'Contact',
        'nav.getStarted': 'Get Started',
        
        // Hero Section
        'hero.badge': '🚀 Revolutionary AI Technology',
        'hero.title': 'Transform Your Farm with AI-Powered Agriculture',
        'hero.description': 'Boost crop yields by 45% with our intelligent farming solutions. Get personalized recommendations, early disease detection, and optimize your resources with cutting-edge AI.',
        'hero.primaryCTA': 'Start Free Trial',
        'hero.secondaryCTA': 'Watch Demo',
        'hero.feature1': 'Free 30-day trial',
        'hero.feature2': '500K+ farmers trust us',
        'hero.feature3': '24/7 expert support',
        'hero.card1.title': '45% Yield Increase',
        'hero.card1.description': 'Average improvement with AI',
        'hero.card2.title': '95% Accuracy',
        'hero.card2.description': 'Disease detection rate',
        'hero.stats.farmers': 'Active Farmers',
        'hero.stats.countries': 'Countries Served',
        'hero.stats.increase': 'Avg. Yield Increase',
        'hero.stats.satisfaction': 'Satisfaction Rate',

        // Features Section
        'features.title': 'Intelligent Features for Modern Farming',
        'features.subtitle': 'Discover how our AI-powered platform revolutionizes every aspect of your agricultural operations',
        'features.smartAdvisory.title': 'Smart Crop Advisory',
        'features.smartAdvisory.description': 'Get personalized recommendations for planting, irrigation, and harvesting based on AI analysis of weather, soil, and crop data.',
        'features.smartAdvisory.feature1': 'Optimal planting schedules',
        'features.smartAdvisory.feature2': 'Water management',
        'features.smartAdvisory.feature3': 'Harvest timing',
        'features.diseaseDetection.title': 'Disease Detection',
        'features.diseaseDetection.description': 'Instantly identify plant diseases and pest infestations using advanced computer vision and machine learning algorithms.',
        'features.diseaseDetection.feature1': 'Real-time scanning',
        'features.diseaseDetection.feature2': '95% accuracy rate',
        'features.diseaseDetection.feature3': 'Treatment recommendations',
        'features.soilAnalysis.title': 'Soil Analysis',
        'features.soilAnalysis.description': 'Comprehensive soil health monitoring with detailed nutrient analysis and customized fertilizer recommendations.',
        'features.soilAnalysis.feature1': 'Nutrient mapping',
        'features.soilAnalysis.feature2': 'pH optimization',
        'features.soilAnalysis.feature3': 'Custom fertilizer blends',
        'features.weatherPrediction.title': 'Weather Prediction',
        'features.weatherPrediction.description': 'Advanced weather forecasting with hyperlocal predictions to help you make informed farming decisions.',
        'features.weatherPrediction.feature1': '7-day forecasts',
        'features.weatherPrediction.feature2': 'Rainfall predictions',
        'features.weatherPrediction.feature3': 'Temperature alerts',
        'features.mobileApp.title': 'Mobile App',
        'features.mobileApp.description': 'Access all features on-the-go with our intuitive mobile application, available for iOS and Android.',
        'features.mobileApp.feature1': 'Offline capabilities',
        'features.mobileApp.feature2': 'Voice commands',
        'features.mobileApp.feature3': 'Real-time notifications',
        'features.multilingual.title': 'Multilingual Support',
        'features.multilingual.description': 'Available in 25+ languages with local farming knowledge and region-specific recommendations.',
        'features.multilingual.feature1': '25+ languages',
        'features.multilingual.feature2': 'Local expertise',
        'features.multilingual.feature3': 'Cultural adaptation',

        // Services Section
        'services.title': 'AI-Powered Agricultural Services',
        'services.subtitle': 'Comprehensive solutions designed to revolutionize your farming experience',
        'services.smartAdvisory.title': 'Smart Crop Advisory',
        'services.smartAdvisory.description': 'AI-driven recommendations for optimal planting, irrigation, and harvesting times based on real-time data analysis.',
        'services.smartAdvisory.feature1': 'Weather prediction',
        'services.smartAdvisory.feature2': 'Soil analysis',
        'services.smartAdvisory.feature3': 'Crop rotation planning',
        'services.smartAdvisory.feature4': 'Yield optimization',
        'services.diseaseDetection.title': 'Disease Detection',
        'services.diseaseDetection.description': 'Advanced computer vision technology to instantly identify plant diseases and pest infestations through image analysis.',
        'services.diseaseDetection.feature1': 'Real-time scanning',
        'services.diseaseDetection.feature2': '95% accuracy rate',
        'services.diseaseDetection.feature3': 'Treatment recommendations',
        'services.diseaseDetection.feature4': 'Progress tracking',
        'services.soilAnalysis.title': 'Soil & Fertilizer Analysis',
        'services.soilAnalysis.description': 'Precision agriculture solutions providing detailed soil composition analysis and customized fertilizer recommendations.',
        'services.soilAnalysis.feature1': 'Nutrient mapping',
        'services.soilAnalysis.feature2': 'pH optimization',
        'services.soilAnalysis.feature3': 'Custom blends',
        'services.soilAnalysis.feature4': 'Cost reduction',
        'services.learnMore': 'Learn More',

        // Testimonials Section
        'testimonials.title': 'Trusted by Farmers Worldwide',
        'testimonials.subtitle': 'See how AgriAI is transforming agriculture across the globe',
        'testimonials.stats.farmers': 'Active Farmers',
        'testimonials.stats.satisfaction': 'Satisfaction Rate',
        'testimonials.stats.countries': 'Countries',
        'testimonials.stats.increase': 'Avg. Yield Increase',
        'testimonials.maria.name': 'Maria Santos',
        'testimonials.maria.title': 'Coffee Farmer',
        'testimonials.maria.location': 'São Paulo, Brazil',
        'testimonials.maria.quote': 'AgriAI helped me increase my coffee yield by 40% while reducing fertilizer costs. The pest detection feature saved my entire crop last season.',
        'testimonials.raj.name': 'Raj Patel',
        'testimonials.raj.title': 'Cotton Farmer',
        'testimonials.raj.location': 'Gujarat, India',
        'testimonials.raj.quote': 'The multilingual support made it easy for me to use. The AI recommendations are spot-on and have helped me optimize water usage significantly.',
        'testimonials.john.name': 'John Mitchell',
        'testimonials.john.title': 'Corn Farmer',
        'testimonials.john.location': 'Iowa, USA',
        'testimonials.john.quote': 'I was skeptical about AI in farming, but AgriAI proved me wrong. The soil analysis and fertilizer recommendations are incredibly accurate.',
        'testimonials.sophie.name': 'Sophie Laurent',
        'testimonials.sophie.title': 'Vineyard Owner',
        'testimonials.sophie.location': 'Provence, France',
        'testimonials.sophie.quote': 'The disease detection feature is a game-changer. We can now identify issues weeks before they become visible to the naked eye.',

        // Contact Section
        'contact.title': 'Get Started with AgriAI',
        'contact.subtitle': 'Ready to transform your farming? Connect with our team of agricultural experts.',
        'contact.form.title': 'Start Your Free Trial',
        'contact.form.description': 'Join thousands of farmers already using AgriAI to optimize their operations.',
        'contact.form.nameLabel': 'Full Name',
        'contact.form.emailLabel': 'Email Address',
        'contact.form.companyLabel': 'Farm/Company Name',
        'contact.form.messageLabel': 'Tell us about your farming needs',
        'contact.form.submitButton': 'Start Free Trial',
        'contact.form.namePlaceholder': 'Enter your full name',
        'contact.form.emailPlaceholder': 'your@email.com',
        'contact.form.companyPlaceholder': 'Your farm or company name',
        'contact.form.messagePlaceholder': 'What crops do you grow? What challenges are you facing?',
        'contact.info.title': 'Contact Information',
        'contact.info.hours': '24/7 Support Available',
        'contact.features.consultation.title': 'Expert Consultation',
        'contact.features.consultation.description': 'Free 30-minute consultation with agricultural specialists',
        'contact.features.community.title': 'Community Access',
        'contact.features.community.description': 'Join our global community of 500K+ farmers',
        'contact.features.security.title': 'Data Security',
        'contact.features.security.description': 'Enterprise-grade security for your farm data',

        // Footer
        'footer.description': 'Revolutionizing agriculture with AI-powered solutions for smarter, more sustainable farming.',
        'footer.product.title': 'Product',
        'footer.product.advisory': 'Crop Advisory',
        'footer.product.detection': 'Disease Detection',
        'footer.product.fertilizer': 'Fertilizer Guidance',
        'footer.product.weather': 'Weather Insights',
        'footer.product.mobile': 'Mobile App',
        'footer.company.title': 'Company',
        'footer.company.about': 'About Us',
        'footer.company.careers': 'Careers',
        'footer.company.press': 'Press',
        'footer.company.blog': 'Blog',
        'footer.company.contact': 'Contact',
        'footer.support.title': 'Support',
        'footer.support.help': 'Help Center',
        'footer.support.docs': 'Documentation',
        'footer.support.api': 'API Reference',
        'footer.support.community': 'Community',
        'footer.support.status': 'Status',
        'footer.newsletter.title': 'Stay Updated',
        'footer.newsletter.description': 'Get the latest agriculture insights and product updates.',
        'footer.newsletter.placeholder': 'Enter your email',
        'footer.newsletter.button': 'Subscribe',
        'footer.copyright': '© 2024 AgriAI. All rights reserved.',
        'footer.legal.privacy': 'Privacy Policy',
        'footer.legal.terms': 'Terms of Service',
        'footer.legal.cookies': 'Cookies'
    },
    es: {
        // Navigation
        'nav.home': 'Inicio',
        'nav.features': 'Características',
        'nav.services': 'Servicios',
        'nav.testimonials': 'Testimonios',
        'nav.contact': 'Contacto',
        'nav.getStarted': 'Comenzar',
        
        // Hero Section
        'hero.badge': '🚀 Tecnología AI Revolucionaria',
        'hero.title': 'Transforma tu Granja con Agricultura Potenciada por AI',
        'hero.description': 'Aumenta el rendimiento de cultivos en un 45% con nuestras soluciones de agricultura inteligente. Obtén recomendaciones personalizadas, detección temprana de enfermedades y optimiza tus recursos con AI de vanguardia.',
        'hero.primaryCTA': 'Iniciar Prueba Gratuita',
        'hero.secondaryCTA': 'Ver Demo',
        'hero.feature1': 'Prueba gratuita de 30 días',
        'hero.feature2': '500K+ agricultores confían en nosotros',
        'hero.feature3': 'Soporte experto 24/7',
        'hero.card1.title': '45% Aumento de Rendimiento',
        'hero.card1.description': 'Mejora promedio con AI',
        'hero.card2.title': '95% Precisión',
        'hero.card2.description': 'Tasa de detección de enfermedades',
        'hero.stats.farmers': 'Agricultores Activos',
        'hero.stats.countries': 'Países Atendidos',
        'hero.stats.increase': 'Aumento Promedio de Rendimiento',
        'hero.stats.satisfaction': 'Tasa de Satisfacción',

        // Add more Spanish translations...
        'contact.form.nameLabel': 'Nombre Completo',
        'contact.form.emailLabel': 'Dirección de Email',
        'contact.form.companyLabel': 'Nombre de Granja/Empresa',
        'contact.form.submitButton': 'Iniciar Prueba Gratuita'
    },
    hi: {
        // Navigation
        'nav.home': 'होम',
        'nav.features': 'विशेषताएं',
        'nav.services': 'सेवाएं',
        'nav.testimonials': 'प्रशंसापत्र',
        'nav.contact': 'संपर्क',
        'nav.getStarted': 'शुरू करें',
        
        // Hero Section
        'hero.badge': '🚀 क्रांतिकारी AI तकनीक',
        'hero.title': 'AI-संचालित कृषि के साथ अपने खेत को बदलें',
        'hero.description': 'हमारे बुद्धिमान कृषि समाधानों के साथ फसल की उपज 45% तक बढ़ाएं। व्यक्तिगत सिफारिशें प्राप्त करें, बीमारी का जल्दी पता लगाएं, और अत्याधुनिक AI के साथ अपने संसाधनों को अनुकूलित करें।',
        'hero.primaryCTA': 'मुफ्त परीक्षण शुरू करें',
        'hero.secondaryCTA': 'डेमो देखें',
        'hero.feature1': '30-दिन का मुफ्त परीक्षण',
        'hero.feature2': '500K+ किसान हम पर भरोसा करते हैं',
        'hero.feature3': '24/7 विशेषज्ञ समर्थन',
        
        // Add more Hindi translations...
        'contact.form.nameLabel': 'पूरा नाम',
        'contact.form.emailLabel': 'ईमेल पता',
        'contact.form.companyLabel': 'खेत/कंपनी का नाम',
        'contact.form.submitButton': 'मुफ्त परीक्षण शुरू करें'
    },
    fr: {
        // Navigation
        'nav.home': 'Accueil',
        'nav.features': 'Fonctionnalités',
        'nav.services': 'Services',
        'nav.testimonials': 'Témoignages',
        'nav.contact': 'Contact',
        'nav.getStarted': 'Commencer',
        
        // Hero Section
        'hero.badge': '🚀 Technologie IA Révolutionnaire',
        'hero.title': 'Transformez votre Ferme avec l\'Agriculture Alimentée par l\'IA',
        'hero.description': 'Augmentez les rendements des cultures de 45% avec nos solutions agricoles intelligentes. Obtenez des recommandations personnalisées, une détection précoce des maladies et optimisez vos ressources avec une IA de pointe.',
        'hero.primaryCTA': 'Commencer l\'Essai Gratuit',
        'hero.secondaryCTA': 'Voir la Démo',
        'hero.feature1': 'Essai gratuit de 30 jours',
        'hero.feature2': '500K+ agriculteurs nous font confiance',
        'hero.feature3': 'Support expert 24/7',
        
        // Add more French translations...
        'contact.form.nameLabel': 'Nom Complet',
        'contact.form.emailLabel': 'Adresse Email',
        'contact.form.companyLabel': 'Nom de Ferme/Entreprise',
        'contact.form.submitButton': 'Commencer l\'Essai Gratuit'
    },
    pt: {
        // Navigation
        'nav.home': 'Início',
        'nav.features': 'Funcionalidades',
        'nav.services': 'Serviços',
        'nav.testimonials': 'Depoimentos',
        'nav.contact': 'Contato',
        'nav.getStarted': 'Começar',
        
        // Hero Section
        'hero.badge': '🚀 Tecnologia IA Revolucionária',
        'hero.title': 'Transforme sua Fazenda com Agricultura Alimentada por IA',
        'hero.description': 'Aumente a produtividade das culturas em 45% com nossas soluções agrícolas inteligentes. Obtenha recomendações personalizadas, detecção precoce de doenças e otimize seus recursos com IA de ponta.',
        'hero.primaryCTA': 'Iniciar Teste Gratuito',
        'hero.secondaryCTA': 'Ver Demo',
        'hero.feature1': 'Teste gratuito de 30 dias',
        'hero.feature2': '500K+ agricultores confiam em nós',
        'hero.feature3': 'Suporte especializado 24/7',
        
        // Add more Portuguese translations...
        'contact.form.nameLabel': 'Nome Completo',
        'contact.form.emailLabel': 'Endereço de Email',
        'contact.form.companyLabel': 'Nome da Fazenda/Empresa',
        'contact.form.submitButton': 'Iniciar Teste Gratuito'
    }
};

// Language flags and names
const languageData = {
    en: { flag: '🇺🇸', name: 'English' },
    es: { flag: '🇪🇸', name: 'Español' },
    hi: { flag: '🇮🇳', name: 'हिंदी' },
    fr: { flag: '🇫🇷', name: 'Français' },
    pt: { flag: '🇧🇷', name: 'Português' }
};

// Current language
let currentLanguage = 'en';

// DOM elements
const languageBtn = document.getElementById('language-btn');
const languageDropdown = document.getElementById('language-dropdown');
const currentLanguageSpan = document.getElementById('current-language');
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const navMenu = document.getElementById('nav-menu');
const header = document.getElementById('header');
const contactForm = document.getElementById('contact-form');
const contactEmail = document.getElementById('contact-email');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeLanguage();
    initializeNavigation();
    initializeScrollEffects();
    initializeAnimations();
    initializeForms();
    updateContactEmail();
});

// Language functions
function initializeLanguage() {
    // Set up language dropdown
    languageBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        languageDropdown.classList.toggle('active');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function() {
        languageDropdown.classList.remove('active');
    });

    // Add language option event listeners
    const languageOptions = document.querySelectorAll('.language-option');
    languageOptions.forEach(option => {
        option.addEventListener('click', function() {
            const lang = this.getAttribute('data-lang');
            changeLanguage(lang);
            languageDropdown.classList.remove('active');
        });
    });

    // Apply initial language
    applyTranslations();
}

function changeLanguage(lang) {
    currentLanguage = lang;
    currentLanguageSpan.textContent = ${languageData[lang].flag} ${languageData[lang].name.split(' ')[0]};
    applyTranslations();
    updateContactEmail();
    localStorage.setItem('selectedLanguage', lang);
}

function applyTranslations() {
    const elements = document.querySelectorAll('[data-key]');
    elements.forEach(element => {
        const key = element.getAttribute('data-key');
        if (translations[currentLanguage] && translations[currentLanguage][key]) {
            element.textContent = translations[currentLanguage][key];
        }
    });

    // Apply placeholder translations
    const placeholderElements = document.querySelectorAll('[data-placeholder-key]');
    placeholderElements.forEach(element => {
        const key = element.getAttribute('data-placeholder-key');
        if (translations[currentLanguage] && translations[currentLanguage][key]) {
            element.placeholder = translations[currentLanguage][key];
        }
    });
}

function updateContactEmail() {
    const emailMap = {
        en: 'hello@agriai.com',
        es: 'hola@agriai.com',
        hi: 'namaste@agriai.com',
        fr: 'bonjour@agriai.com',
        pt: 'ola@agriai.com'
    };
    
    if (contactEmail) {
        contactEmail.textContent = emailMap[currentLanguage] || emailMap.en;
    }
}

// Navigation functions
function initializeNavigation() {
    // Mobile menu toggle
    mobileMenuBtn.addEventListener('click', function() {
        navMenu.classList.toggle('active');
    });

    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const headerHeight = header.offsetHeight;
                const targetPosition = targetSection.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
            
            // Close mobile menu
            navMenu.classList.remove('active');
        });
    });

    // Active navigation highlighting
    window.addEventListener('scroll', updateActiveNavigation);
}

function updateActiveNavigation() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    const headerHeight = header.offsetHeight;
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - headerHeight - 100;
        const sectionHeight = section.offsetHeight;
        
        if (window.pageYOffset >= sectionTop && 
            window.pageYOffset < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === #${current}) {
            link.classList.add('active');
        }
    });
}

// Scroll effects
function initializeScrollEffects() {
    window.addEventListener('scroll', function() {
        // Header scroll effect
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

// Animations
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.feature-card, .service-item, .testimonial-card, .stat-item');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Forms
function initializeForms() {
    // Contact form
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleContactSubmission();
        });
    }

    // Newsletter form
    const newsletterForm = document.querySelector('.newsletter-form');
    if (newsletterForm) {
        newsletterForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleNewsletterSubmission();
        });
    }
}

function handleContactSubmission() {
    const formData = new FormData(contactForm);
    const submitButton = contactForm.querySelector('button[type="submit"]');
    const originalText = submitButton.innerHTML;
    
    // Show loading state
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    submitButton.disabled = true;
    
    // Simulate form submission
    setTimeout(() => {
        // Show success message
        showNotification('Thank you! We\'ll contact you soon.', 'success');
        contactForm.reset();
        
        // Reset button
        submitButton.innerHTML = originalText;
        submitButton.disabled = false;
    }, 2000);
}

function handleNewsletterSubmission() {
    const newsletterForm = document.querySelector('.newsletter-form');
    const emailInput = newsletterForm.querySelector('input[type="email"]');
    const submitButton = newsletterForm.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    // Show loading state
    submitButton.textContent = 'Subscribing...';
    submitButton.disabled = true;
    
    // Simulate subscription
    setTimeout(() => {
        showNotification('Successfully subscribed to newsletter!', 'success');
        emailInput.value = '';
        
        // Reset button
        submitButton.textContent = originalText;
        submitButton.disabled = false;
    }, 1500);
}

// Utility functions
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = notification notification-${type};
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles for notification
    const styles = `
        .notification {
            position: fixed;
            top: 100px;
            right: 20px;
            background: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            border-left: 4px solid var(--primary-color);
        }
        
        .notification.show {
            transform: translateX(0);
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .notification i {
            color: var(--primary-color);
        }
        
        .notification-success {
            border-left-color: #10b981;
        }
        
        .notification-success i {
            color: #10b981;
        }
    `;
    
    // Add styles to head if not already added
    if (!document.getElementById('notification-styles')) {
        const styleSheet = document.createElement('style');
        styleSheet.id = 'notification-styles';
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }
    
    // Add to body
    document.body.appendChild(notification);
    
    // Show notification
    setTimeout(() => notification.classList.add('show'), 100);
    
    // Remove notification after 4 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Button click handlers
document.addEventListener('click', function(e) {
    // Handle CTA buttons
    if (e.target.matches('.btn') || e.target.closest('.btn')) {
        const button = e.target.matches('.btn') ? e.target : e.target.closest('.btn');
        const buttonText = button.textContent.trim();
        
        if (buttonText.includes('Trial') || buttonText.includes('Started')) {
            // Scroll to contact form
            document.getElementById('contact').scrollIntoView({ behavior: 'smooth' });
        } else if (buttonText.includes('Demo')) {
            showNotification('Demo video will be available soon!', 'info');
        } else if (buttonText.includes('Learn More')) {
            showNotification('More details coming soon!', 'info');
        }
    }
});

// Load saved language preference
window.addEventListener('load', function() {
    const savedLanguage = localStorage.getItem('selectedLanguage');
    if (savedLanguage && languageData[savedLanguage]) {
        changeLanguage(savedLanguage);
    }
});

// Add scroll-to-top functionality
const scrollToTopBtn = document.createElement('button');
scrollToTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
scrollToTopBtn.className = 'scroll-to-top';
scrollToTopBtn.style.cssText = `
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    cursor: pointer;
    box-shadow: var(--shadow-lg);
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    z-index: 1000;
`;

document.body.appendChild(scrollToTopBtn);

window.addEventListener('scroll', function() {
    if (window.pageYOffset > 300) {
        scrollToTopBtn.style.opacity = '1';
        scrollToTopBtn.style.visibility = 'visible';
    } else {
        scrollToTopBtn.style.opacity = '0';
        scrollToTopBtn.style.visibility = 'hidden';
    }
});

scrollToTopBtn.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Add mouse cursor effects for interactive elements
document.addEventListener('mousemove', function(e) {
    const interactiveElements = document.querySelectorAll('.btn, .nav-link, .social-link, .language-option');
    
    interactiveElements.forEach(element => {
        const rect = element.getBoundingClientRect();
        const isHovering = e.clientX >= rect.left && e.clientX <= rect.right && 
                          e.clientY >= rect.top && e.clientY <= rect.bottom;
        
        if (isHovering) {
            document.body.style.cursor = 'pointer';
        }
    });
});

// Performance optimization: Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debouncing to scroll handlers
const debouncedScrollHandler = debounce(function() {
    updateActiveNavigation();
}, 10);

window.addEventListener('scroll', debouncedScrollHandler);