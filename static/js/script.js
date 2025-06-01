document.getElementById('uploadForm').addEventListener('submit', async function(event) {
  event.preventDefault();
  const formData = new FormData();
  const fileInput = document.getElementById('file');
  formData.append('file', fileInput.files[0]);

  const response = await fetch('/upload', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  const imagePreview = document.getElementById('imagePreview');
  const resultText = document.getElementById('resultText');

  if (fileInput.files[0]) {
    imagePreview.src = URL.createObjectURL(fileInput.files[0]);
    imagePreview.style.display = 'block';
  }

  resultText.innerText = `Detected Plate: ${result.plate_text}`;
});