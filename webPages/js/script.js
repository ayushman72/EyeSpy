function formatDate(date) {
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

  const dayOfWeek = days[date.getDay()];
  const dayOfMonth = date.getDate();
  const month = months[date.getMonth()];
  const year = date.getFullYear();

  return `${dayOfWeek}, ${dayOfMonth < 10 ? '0' : ''}${dayOfMonth} ${month} ${year}`;
}

function formatTime(date) {
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const seconds = date.getSeconds();

  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hourFormat = hours % 12 || 12; // Convert hour to 12-hour format

  return `${hourFormat < 10 ? '0' : ''}${hourFormat}:${minutes < 10 ? '0' : ''}${minutes}:${seconds < 10 ? '0' : ''}${seconds} ${ampm}`;
}

function updateDateTime() {
  var currentDate = new Date();
  var formattedDate = formatDate(currentDate);
  var formattedTime = formatTime(currentDate);

  document.querySelector('.date').textContent = formattedDate;
  document.querySelector('.time').textContent = formattedTime;

  // Check for reminder notification
  checkForReminderNotification();
}

// Function to check for reminder notification
function checkForReminderNotification() {
  const reminderNotification = localStorage.getItem('reminderNotification');
  if (reminderNotification === 'true') {
    // Display the notification
    showNotification();
    // Clear the flag
    localStorage.removeItem('reminderNotification');
  }
}

// Function to show notification
function showNotification() {
  // Display the toast notification using Toastify
  Toastify({
    text: 'Reminder notification!',
    duration: 50000, 
    close: true,
    gravity: 'top',
    position: 'center',
    backgroundColor: 'white', // Change background color to white
    color: 'black', // Change text color to black
    width: "400px", // Increase width of notification
    stopOnFocus: true,
    style: {
      // Add custom styles for centering text and increasing height
      'text-align': 'center',
      'height': '100px', // Increase height
      'display': 'flex',
      'justify-content': 'center', // Horizontally center the content
      'align-items': 'center',
      'margin-top':'0px',
      'font-size':'25px'


    },
    onClick: function() {
      // Hide notification when clicked
      this.hide();
    }
  }).showToast();
}



// Update date and time every second
setInterval(updateDateTime, 1000);

// Initial update
updateDateTime();
