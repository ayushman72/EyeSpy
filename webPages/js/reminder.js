document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('reminderForm');
  const reminderList = document.getElementById('reminders');
  const notification = document.getElementById('notification');

  form.addEventListener('submit', function(event) {
    event.preventDefault();

    const reminderInput = document.getElementById('reminder');
    const dateInput = document.getElementById('date');
    const timeInput = document.getElementById('time');

    const reminder = reminderInput.value;
    const date = dateInput.value;
    const time = timeInput.value;

    const dateTime = new Date(`${date}T${time}`);
    const currentTime = new Date();

    if (dateTime > currentTime) {
      const timeDiff = dateTime - currentTime;

      setTimeout(() => {
        // Remove the expired reminder from the list
        reminderList.querySelectorAll('li').forEach(function(item) {
          if (item.textContent.includes(reminder)) {
            item.remove();
          }
        });

        // Show toast notification
        notification.textContent = `Reminder: ${reminder} Click here to Close`;
        notification.classList.add('show');

        // Store flag in localStorage
        localStorage.setItem('reminderNotification', 'true');

        // Hide notification after 100 seconds
        setTimeout(() => {
          notification.classList.remove('show');
        }, 100000);
      }, timeDiff);
    }

    // Create reminder item
    const li = document.createElement('li');
    li.textContent = `${reminder} - ${dateTime.toLocaleString()}`;
    reminderList.appendChild(li);

    // Clear input fields
    reminderInput.value = '';
    dateInput.value = '';
    timeInput.value = '';
  });

  // Close notification when close button is clicked
  notification.addEventListener('click', function() {
    notification.classList.remove('show');
  });
});
