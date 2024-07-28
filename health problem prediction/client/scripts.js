document.addEventListener('DOMContentLoaded', function () {
    const dropdownMenu = document.getElementById('dropdown-menu');
    const predictButton = document.getElementById('predict-button');
    const dropdownButton = document.querySelector('.dropbtn');
    const dropdown = document.querySelector('.dropdown');

    if (!dropdownMenu || !predictButton || !dropdownButton) {
        console.error('Required elements not found!');
        return;
    }

    fetch('http://127.0.0.1:5000/get_symptoms')
        .then(response => response.json())
        .then(data => {
            data.symptoms.forEach(symptom => {
                const label = document.createElement('label');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.value = symptom;
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(symptom));
                dropdownMenu.appendChild(label);
            });
        })
        .catch(error => console.error('Error fetching symptoms:', error));

    dropdownButton.addEventListener('click', function() {
        dropdown.classList.toggle('show');
    });

    predictButton.addEventListener('click', predictHealthProblem);

    function predictHealthProblem() {
        const selectedSymptoms = Array.from(document.querySelectorAll('#dropdown-menu input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);

        fetch('http://127.0.0.1:5000/predict_problem', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symptoms: selectedSymptoms })
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('health-problem').textContent = data.health_problem || 'No health problem found';
            document.getElementById('precautions').textContent = data.precaution || 'No precautions available';
            document.getElementById('home-remedies').textContent = data.home_remedies || 'No home remedies available';
        })
        .catch(error => console.error('Error predicting health problem:', error));
    }
});
