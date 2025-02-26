function displayResponse(data) {
    const responseElement = document.getElementById('response');
    if (data.isHtml) {
        responseElement.innerHTML = data.response;
    } else {
        responseElement.textContent = data.response;
    }
} 