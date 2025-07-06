        document.addEventListener('DOMContentLoaded', () => {
            // --- DOM Element References ---
            const urlInput = document.getElementById('url-input');
            const analyzeBtn = document.getElementById('analyze-btn');
            const modelSelect = document.getElementById('model-select');
            const inputGroup = document.getElementById('input-group');
            const loader = document.getElementById('loader');
            const loaderText = document.getElementById('loader-text');
            const errorContainer = document.getElementById('error-container');
            const resultsSection = document.getElementById('results-section');
            const reviewsContainer = document.getElementById('reviews-container');

            // Summary Stat Elements
            const totalReviewsEl = document.getElementById('total-reviews');
            const primaryRealReviewsEl = document.getElementById('primary-real-reviews');
            const primaryFakeReviewsEl = document.getElementById('primary-fake-reviews');
            const primaryRealLabelEl = document.getElementById('primary-real-label');
            const primaryFakeLabelEl = document.getElementById('primary-fake-label');
            const trustScoreEl = document.getElementById('trust-score');
            
            // Model comparison elements
            const svmRealCountEl = document.getElementById('svm-real-count');
            const svmRealPercentEl = document.getElementById('svm-real-percent');
            const svmFakeCountEl = document.getElementById('svm-fake-count');
            const svmFakePercentEl = document.getElementById('svm-fake-percent');
            
            const rfRealCountEl = document.getElementById('rf-real-count');
            const rfRealPercentEl = document.getElementById('rf-real-percent');
            const rfFakeCountEl = document.getElementById('rf-fake-count');
            const rfFakePercentEl = document.getElementById('rf-fake-percent');
            
            // Filter and Sort Elements
            const filterBtns = document.querySelectorAll('.filter-btn');
            const sortSelect = document.getElementById('sort-select');
            
            // Modal Elements
            const modal = document.getElementById('how-it-works-modal');
            const modalBtn = document.getElementById('how-it-works-btn');
            const closeBtn = document.querySelector('.close-btn');

            // Chart instances
            let compositionChart = null;
            let sentimentChart = null;
            
            // Data storage
            let allReviews = [];
            let filteredReviews = [];

            // --- Event Listeners ---
            analyzeBtn.addEventListener('click', analyzeReviews);
            
            modelSelect.addEventListener('change', function() {
                // Add visual feedback for model selection
                const selectedOption = this.options[this.selectedIndex];
                this.style.borderColor = 'var(--primary-accent)';
                setTimeout(() => {
                    this.style.borderColor = 'var(--border-color)';
                }, 500);
            });
            
            // URL validation
            urlInput.addEventListener('change', validateUrlRealtime);
            urlInput.addEventListener('input', debounce(validateUrlRealtime, 300));
            
            // Filter and Sort
            filterBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    filterBtns.forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    filterReviews(btn.dataset.filter);
                });
            });
            
            sortSelect.addEventListener('change', () => {
                sortReviews(sortSelect.value);
            });
            
            // Modal listeners
            modalBtn.addEventListener('click', () => {
                modal.style.display = 'flex';
                modal.setAttribute('aria-hidden', 'false');
            });
            
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
                modal.setAttribute('aria-hidden', 'true');
            });
            
            window.addEventListener('click', (event) => {
                if (event.target == modal) {
                    modal.style.display = 'none';
                    modal.setAttribute('aria-hidden', 'true');
                }
            });
            
            // Keyboard navigation for modal
            document.addEventListener('keydown', (event) => {
                if (event.key === 'Escape' && modal.style.display === 'flex') {
                    modal.style.display = 'none';
                    modal.setAttribute('aria-hidden', 'true');
                }
            });

            // --- Utility Functions ---
            
            /**
             * Debounce function to limit the rate of function calls
             */
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

            /**
             * Validates the URL and provides visual feedback.
             */
            function validateUrlRealtime() {
                const url = urlInput.value.trim();
                inputGroup.classList.remove('valid', 'invalid');
                if (url) {
                    // This is a safer regex that is less likely to cause browser hanging (catastrophic backtracking).
                    const urlPattern = /^(https?:\/\/)?(([\w-]+\.)+)([a-z]{2,6})(\/[\w-./?%&=]*)?$/i;
                    if (urlPattern.test(url)) {
                        inputGroup.classList.add('valid');
                    } else {
                        inputGroup.classList.add('invalid');
                    }
                }
            }

            /**
             * Updates loader steps with animation
             */
            function updateLoaderStep(stepIndex) {
                const steps = document.querySelectorAll('.loader-steps .step');
                steps.forEach((step, index) => {
                    step.classList.remove('active');
                    if (index <= stepIndex) {
                        step.classList.add('active');
                    }
                });
            }

            /**
             * Main function to trigger the review analysis process.
             */
            function analyzeReviews() {
                const url = urlInput.value.trim();
                const selectedModel = modelSelect.value;
                
                if (!url) {
                    showError("Please enter a product URL to analyze.");
                    return;
                }

                // Reset UI state
                resetUI();
                loader.style.display = 'flex';
                
                // Initialize loader steps
                updateLoaderStep(0);
                
                const loaderStates = [
                    "Initializing AI models...",
                    "Scraping product page...", 
                    "Analyzing review text...", 
                    "Extracting sentiment...", 
                    "Running AI prediction models...", 
                    "Compiling results..."
                ];
                
                let stateIndex = 0;
                loaderText.textContent = loaderStates[0];
                
                const loaderInterval = setInterval(() => {
                    stateIndex = (stateIndex + 1) % loaderStates.length;
                    loaderText.textContent = loaderStates[stateIndex];
                    
                    // Update step indicator
                    const stepIndex = Math.floor((stateIndex / loaderStates.length) * 4);
                    updateLoaderStep(stepIndex);
                }, 2000);

                // Fetch data from backend
                fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        url: url,
                        selectedModel: selectedModel
                    })
                })
                .then(response => {
                    clearInterval(loaderInterval);
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.error || 'Failed to analyze reviews');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    loader.style.display = 'none';
                    displayResults(data);
                })
                .catch(error => {
                    clearInterval(loaderInterval);
                    loader.style.display = 'none';
                    showError(error.message);
                });
            }

            /**
             * Displays the analysis results.
             * @param {Object} data - The analysis results data.
             */
            function displayResults(data) {
                resultsSection.style.display = 'block';
                displaySummaryStats(data.summary);
                displayReviewCards(data.reviews);
                renderCharts(data.summary);
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }

            /**
             * Displays the summary statistics.
             * @param {Object} summary - The summary statistics.
             */
            function displaySummaryStats(summary) {
                // Update summary stats with animations
                animateNumber(totalReviewsEl, summary.totalReviews);
                animateNumber(primaryRealReviewsEl, summary.primary.realCount);
                animateNumber(primaryFakeReviewsEl, summary.primary.fakeCount);
                animateNumber(trustScoreEl, summary.trustScore, true, '%');
                
                // Hide all model cards first
                const modelCards = document.querySelectorAll('.model-stat-card');
                modelCards.forEach(card => {
                    card.style.display = 'none';
                });
                
                // Show only the selected model card
                if (summary.selectedModel === 'svm') {
                    modelCards[0].style.display = 'block'; // SVM card
                    // Update SVM stats
                    animateNumber(svmRealCountEl, summary.svm.realCount);
                    animateNumber(svmRealPercentEl, summary.svm.realPercentage, true, '%');
                    animateNumber(svmFakeCountEl, summary.svm.fakeCount);
                    animateNumber(svmFakePercentEl, summary.svm.fakePercentage, true, '%');
                    // Update labels
                    primaryRealLabelEl.textContent = 'SVM Authentic Reviews';
                    primaryFakeLabelEl.textContent = 'SVM Likely Fake Reviews';
                } else if (summary.selectedModel === 'rf') {
                    modelCards[1].style.display = 'block'; // Random Forest card
                    // Update Random Forest stats
                    animateNumber(rfRealCountEl, summary.randomForest.realCount);
                    animateNumber(rfRealPercentEl, summary.randomForest.realPercentage, true, '%');
                    animateNumber(rfFakeCountEl, summary.randomForest.fakeCount);
                    animateNumber(rfFakePercentEl, summary.randomForest.fakePercentage, true, '%');
                    // Update labels
                    primaryRealLabelEl.textContent = 'RF Authentic Reviews';
                    primaryFakeLabelEl.textContent = 'RF Likely Fake Reviews';
                }
            }

            /**
             * Animates number changes for a smooth visual effect.
             * @param {HTMLElement} element - The element to animate.
             * @param {number} targetValue - The target value to animate to.
             * @param {boolean} percentage - Whether this is a percentage value.
             * @param {string} suffix - Optional suffix to append.
             */
            function animateNumber(element, targetValue, percentage = null, suffix = '') {
                const startValue = 0;
                const duration = 1000;
                const startTime = performance.now();
                
                function updateNumber(currentTime) {
                    const elapsed = currentTime - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    // Easing function for smooth animation
                    const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                    const currentValue = startValue + (targetValue - startValue) * easeOutQuart;
                    
                    if (percentage) {
                        element.textContent = Math.round(currentValue) + suffix;
                    } else {
                        element.textContent = Math.round(currentValue);
                    }
                    
                    if (progress < 1) {
                        requestAnimationFrame(updateNumber);
                    }
                }
                
                requestAnimationFrame(updateNumber);
            }

            /**
             * Renders the charts for data visualization.
             * @param {Object} summary - The summary statistics.
             */
            function renderCharts(summary) {
                // Destroy existing charts if they exist
                if (compositionChart) {
                    compositionChart.destroy();
                }
                if (sentimentChart) {
                    sentimentChart.destroy();
                }

                // Composition Chart
                const compositionCtx = document.getElementById('composition-chart').getContext('2d');
                compositionChart = new Chart(compositionCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Authentic Reviews', 'Likely Fake Reviews'],
                        datasets: [{
                            data: [summary.primary.realCount, summary.primary.fakeCount],
                            backgroundColor: [
                                'rgba(34, 197, 94, 0.8)',
                                'rgba(239, 68, 68, 0.8)'
                            ],
                            borderColor: [
                                'rgba(34, 197, 94, 1)',
                                'rgba(239, 68, 68, 1)'
                            ],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: {
                                    color: '#CBD5E1',
                                    padding: 20,
                                    usePointStyle: true
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                                titleColor: '#F8FAFC',
                                bodyColor: '#CBD5E1',
                                borderColor: '#475569',
                                borderWidth: 1
                            }
                        }
                    }
                });

                // Sentiment Chart
                const sentimentCtx = document.getElementById('sentiment-chart').getContext('2d');
                sentimentChart = new Chart(sentimentCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Positive', 'Neutral', 'Negative'],
                        datasets: [{
                            label: 'Sentiment Distribution',
                            data: [
                                summary.sentimentCounts.Positive || 0,
                                summary.sentimentCounts.Neutral || 0,
                                summary.sentimentCounts.Negative || 0
                            ],
                            backgroundColor: [
                                'rgba(34, 197, 94, 0.8)',
                                'rgba(107, 114, 128, 0.8)',
                                'rgba(239, 68, 68, 0.8)'
                            ],
                            borderColor: [
                                'rgba(34, 197, 94, 1)',
                                'rgba(107, 114, 128, 1)',
                                'rgba(239, 68, 68, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: {
                                    color: 'rgba(71, 85, 105, 0.2)'
                                },
                                ticks: {
                                    color: '#CBD5E1'
                                }
                            },
                            x: {
                                grid: {
                                    color: 'rgba(71, 85, 105, 0.2)'
                                },
                                ticks: {
                                    color: '#CBD5E1'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                                titleColor: '#F8FAFC',
                                bodyColor: '#CBD5E1',
                                borderColor: '#475569',
                                borderWidth: 1
                            }
                        }
                    }
                });
            }

            /**
             * Displays individual review cards.
             * @param {Array} reviews - Array of review objects.
             */
            function displayReviewCards(reviews) {
                allReviews = reviews;
                filteredReviews = [...reviews];
                
                renderReviewCards();
            }

            /**
             * Renders the review cards based on current filter and sort
             */
            function renderReviewCards() {
                reviewsContainer.innerHTML = '';
                
                if (filteredReviews.length === 0) {
                    reviewsContainer.innerHTML = `
                        <div class="no-reviews">
                            <i class="bi bi-inbox"></i>
                            <p>No reviews match the current filter criteria.</p>
                        </div>
                    `;
                    return;
                }
                
                filteredReviews.forEach((review, index) => {
                    const card = document.createElement('div');
                    card.className = `review-card ${review.primary_prediction.toLowerCase()}`;
                    card.style.animationDelay = `${index * 0.1}s`;
                    
                    const sentimentClass = review.sentiment.toLowerCase();
                    const sentimentBarWidth = review.sentiment === 'Positive' ? '100%' : 
                                            review.sentiment === 'Negative' ? '100%' : '50%';
                    
                    card.innerHTML = `
                        <div class="review-header">
                            <div class="icon ${review.primary_prediction.toLowerCase()}">
                                ${review.primary_prediction === 'Real' ? '✓' : '⚠'}
                            </div>
                            <h3>${review.primary_prediction === 'Real' ? 'Authentic Review' : 'Likely Fake Review'}</h3>
                        </div>
                        <div class="review-body">
                            "${review.reviewText}"
                        </div>
                        <div class="review-footer">
                            <div class="rating">
                                <i class="bi bi-star-fill"></i>
                                ${review.rating}/5
                            </div>
                            <div class="sentiment-bar">
                                <span>${review.sentiment}</span>
                                <div class="sentiment-bar-inner ${sentimentClass}" style="width: ${sentimentBarWidth}"></div>
                            </div>
                        </div>
                    `;
                    
                    reviewsContainer.appendChild(card);
                });
            }

            /**
             * Filters reviews based on selected filter
             */
            function filterReviews(filter) {
                switch(filter) {
                    case 'real':
                        filteredReviews = allReviews.filter(review => review.primary_prediction === 'Real');
                        break;
                    case 'fake':
                        filteredReviews = allReviews.filter(review => review.primary_prediction === 'Fake');
                        break;
                    default:
                        filteredReviews = [...allReviews];
                }
                renderReviewCards();
            }

            /**
             * Sorts reviews based on selected sort option
             */
            function sortReviews(sortOption) {
                switch(sortOption) {
                    case 'rating-high':
                        filteredReviews.sort((a, b) => b.rating - a.rating);
                        break;
                    case 'rating-low':
                        filteredReviews.sort((a, b) => a.rating - b.rating);
                        break;
                    case 'sentiment':
                        const sentimentOrder = { 'Positive': 3, 'Neutral': 2, 'Negative': 1 };
                        filteredReviews.sort((a, b) => sentimentOrder[b.sentiment] - sentimentOrder[a.sentiment]);
                        break;
                    default:
                        // Keep original order
                        break;
                }
                renderReviewCards();
            }

            /**
             * Resets the UI to its initial state.
             */
            function resetUI() {
                errorContainer.style.display = 'none';
                resultsSection.style.display = 'none';
                reviewsContainer.innerHTML = '';
                
                // Reset filter and sort
                filterBtns.forEach(btn => btn.classList.remove('active'));
                filterBtns[0].classList.add('active'); // "All Reviews" button
                sortSelect.value = 'default';
                
                // Reset loader steps
                updateLoaderStep(-1);
            }

            /**
             * Shows an error message to the user.
             * @param {string} message - The error message to display.
             */
            function showError(message) {
                errorContainer.textContent = message;
                errorContainer.style.display = 'block';
                errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Auto-hide error after 5 seconds
                setTimeout(() => {
                    errorContainer.style.display = 'none';
                }, 5000);
            }
        });