document.addEventListener('DOMContentLoaded', () => {
    const symptomsContainer = document.getElementById('symptoms-container');
    const searchInput = document.getElementById('search-input');
    const predictButton = document.getElementById('predict-button');
    const healthProblemElement = document.getElementById('health-problem');
    const precautionsElement = document.getElementById('precautions');
    const homeRemediesElement = document.getElementById('home-remedies');
    const selectedSymptomsElement = document.getElementById('selected-symptoms');

    const apiBaseUrl = 'http://localhost:5000';
    let symptoms = [];

    // Fetch symptoms and populate dropdown
    fetch(`${apiBaseUrl}/get_symptoms`)
        .then(response => response.json())
        .then(data => {
            symptoms = data.symptoms;
            populateSymptoms(symptoms);
        });

    function populateSymptoms(symptoms) {
        symptomsContainer.innerHTML = ''; // Clear existing content
        symptoms.forEach(symptom => {
            const checkboxContainer = document.createElement('div');
            checkboxContainer.classList.add('checkbox-container');
            checkboxContainer.innerHTML = `
                <input type="checkbox" value="${symptom}" id="${symptom}">
                <label for="${symptom}">${symptom}</label>
            `;
            symptomsContainer.appendChild(checkboxContainer);
        });
    }

    function filterSymptoms() {
        const query = searchInput.value.toLowerCase();
        const allCheckboxes = document.querySelectorAll('#symptoms-container .checkbox-container');
        
        allCheckboxes.forEach(container => {
            const label = container.querySelector('label');
            const symptom = label.textContent.toLowerCase();
            if (symptom.includes(query)) {
                label.classList.add('highlight');
                container.style.display = 'block';
            } else {
                label.classList.remove('highlight');
                container.style.display = 'none';
            }
        });
    }

    function predictHealthProblem() {
        const selectedCheckboxes = document.querySelectorAll('#symptoms-container input[type="checkbox"]:checked');
        const selectedSymptoms = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
        
        // Display selected symptoms
        selectedSymptomsElement.textContent = `Selected symptoms are: ${selectedSymptoms.join(', ')}`;

        // Fetch prediction
        fetch(`${apiBaseUrl}/predict_problem`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symptoms: selectedSymptoms })
        })
        .then(response => response.json())
        .then(data => {
            healthProblemElement.textContent = data.health_problem;
            precautionsElement.textContent = data.precaution;
            homeRemediesElement.textContent = data.home_remedies;
        });
    }

    searchInput.addEventListener('input', filterSymptoms);
    predictButton.addEventListener('click', predictHealthProblem);
});
