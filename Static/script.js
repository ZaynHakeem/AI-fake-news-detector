function checkNews() {
    const article = document.getElementById('article').value;
    const result = document.getElementById('result');
    const button = document.querySelector('button');

    result.textContent = 'Analyzing...';
    result.classList.remove('show');
    button.disabled = true;
    button.textContent = 'Checking...';

    if (!article.trim()) {
        result.textContent = 'Please enter some news to analyze!';
        result.classList.add('show');
        button.disabled = false;
        button.textContent = 'Check It!';
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ article: article })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Server error');
        }
        return response.json();
    })
    .then(data => {
        result.textContent = data.prediction;
        result.style.color = data.prediction.includes('Fake') ? '#e74c3c' : '#2ecc71';
        result.classList.add('show');
    })
    .catch(error => {
        console.error('Error:', error);
        result.textContent = 'Oops! Couldn’t analyze that. Try again?';
        result.classList.add('show');
    })
    .finally(() => {
        button.disabled = false;
        button.textContent = 'Check It!';
    });
}