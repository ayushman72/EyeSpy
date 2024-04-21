const cityInput = document.querySelector(".city-input");
const searchButton = document.querySelector(".search-btn");
const locationButton = document.querySelector(".location-btn");
const currentAQIDiv = document.querySelector(".current-aqi");
const aqiCardsDiv = document.querySelector(".aqi-cards");

const API_KEY = "447545d7ef0bfacfe791012707fed2a3";

const createAQICard = (cityName, aqiItem, index) => {
    if(index === 0) { // HTML for the main AQI card
        return `<div class="details">
                    <h2>${cityName} (${new Date().toLocaleDateString()})</h2>
                    <h6>AQI: ${aqiItem.main.aqi}</h6>
                    <h6>PM2.5: ${aqiItem.components.pm2_5} μg/m³</h6>
                    <h6>PM10: ${aqiItem.components.pm10} μg/m³</h6>
                </div>`;
    } else { // HTML for the other AQI forecast cards
        return `<li class="card">
                    <h3>${new Date(aqiItem.dt * 1000).toLocaleDateString()}</h3>
                    <h6>AQI: ${aqiItem.main.aqi}</h6>
                    <h6>PM2.5: ${aqiItem.components.pm2_5} μg/m³</h6>
                    <h6>PM10: ${aqiItem.components.pm10} μg/m³</h6>
                </li>`;
    }
}

const getAQIDetails = (cityName, latitude, longitude) => {
    const AQI_API_URL = `http://api.openweathermap.org/data/2.5/air_pollution?lat=${latitude}&lon=${longitude}&appid=${API_KEY}`;

    fetch(AQI_API_URL)
        .then(response => response.json())
        .then(data => {
            // Clear previous AQI data
            currentAQIDiv.innerHTML = "";
            aqiCardsDiv.innerHTML = "";

            // Create AQI cards and add them to the DOM
            data.list.forEach((aqiItem, index) => {
                const html = createAQICard(cityName, aqiItem, index);
                if (index === 0) {
                    currentAQIDiv.insertAdjacentHTML("beforeend", html);
                } else {
                    aqiCardsDiv.insertAdjacentHTML("beforeend", html);
                }
            });

            // Update current AQI value
            const currentAQI = data.list[0].main.aqi;
            currentAQIDiv.querySelector("h6:nth-of-type(1)").textContent = `AQI: ${currentAQI}`;
        })
        .catch(error => {
            console.error("Error fetching AQI data:", error);
            alert("An error occurred while fetching AQI data!");
        });
}

const getCityCoordinates = () => {
    const cityName = cityInput.value.trim();
    if (cityName === "") return;
    const GEO_API_URL = `http://api.openweathermap.org/geo/1.0/direct?q=${cityName}&limit=1&appid=${API_KEY}`;

    // Get city coordinates from the API response
    fetch(GEO_API_URL)
        .then(response => response.json())
        .then(data => {
            if (!data.length) return alert(`No coordinates found for ${cityName}`);
            const { lat, lon } = data[0];
            getAQIDetails(cityName, lat, lon);
        })
        .catch(() => {
            alert("An error occurred while fetching the coordinates!");
        });
}

const getUserCoordinates = () => {
    navigator.geolocation.getCurrentPosition(
        position => {
            const { latitude, longitude } = position.coords;
            getAQIDetails("Your Location", latitude, longitude);
        },
        error => {
            if (error.code === error.PERMISSION_DENIED) {
                alert("Geolocation request denied. Please reset location permission to grant access again.");
            } else {
                alert("Geolocation request error. Please reset location permission.");
            }
        }
    );
}

const DEFAULT_CITY_NAME = "Ranchi";
const DEFAULT_CITY_LAT = 23.3441; // New York's latitude
const DEFAULT_CITY_LON = 85.3096; // New York's longitude

const getDefaultCityAQI = () => {
    getAQIDetails(DEFAULT_CITY_NAME, DEFAULT_CITY_LAT, DEFAULT_CITY_LON);
}

window.addEventListener("load", getDefaultCityAQI);

locationButton.addEventListener("click", getUserCoordinates);
searchButton.addEventListener("click", getCityCoordinates);
cityInput.addEventListener("keyup", e => {
    if (e.key === "Enter") {
        getCityCoordinates();
    }
});
