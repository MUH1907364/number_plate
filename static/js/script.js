const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const canvas = document.getElementById('resultCanvas');
const ctx = canvas.getContext('2d');
const resultText = document.getElementById('resultText');
const previewContainer = document.getElementById('previewContainer');
const loader = document.getElementById('loader');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  loader.classList.remove('hidden');
  previewContainer.classList.add('hidden');

  const response = await fetch('/upload', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  loader.classList.add('hidden');

  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    if (data.box) {
      const [x1, y1, x2, y2] = data.box;
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.font = 'bold 25px Arial';
      ctx.fillStyle = 'yellow';
      ctx.fillText(data.plate_text, x1, y1 - 10);
    }

    resultText.textContent = `Detected Plate: ${data.plate_text}`;
    previewContainer.classList.remove('hidden');
  };
  img.src = URL.createObjectURL(file);
});
