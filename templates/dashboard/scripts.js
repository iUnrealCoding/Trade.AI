document.addEventListener('DOMContentLoaded', () => {
    // Initialize your chart here using your preferred library, e.g., Chart.js, D3.js, etc.
    // For example, using Chart.js:
    const ctx = document.getElementById('chart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['8:00', '10:00', '12:00', '14:00', '16:00'],
            datasets: [{
                label: 'S&P 500',
                data: [0.062, 0.063, 0.065, 0.064, 0.066],
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});
