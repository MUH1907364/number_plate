// Show image preview before form submission
document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('file');
  const previewContainer = document.createElement('div');
  previewContainer.id = 'preview-container';
  previewContainer.style.marginTop = '15px';

  const form = document.querySelector('form');
  form.appendChild(previewContainer);

  fileInput.addEventListener('change', function (event) {
    const file = event.target.files[0];
    const preview = document.createElement('img');

    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = function (e) {
        preview.src = e.target.result;
        preview.style.maxWidth = '100%';
        preview.style.border = '1px solid #ccc';
        preview.style.marginTop = '10px';

        previewContainer.innerHTML = '';
        previewContainer.appendChild(preview);
      };
      reader.readAsDataURL(file);
    } else {
      previewContainer.innerHTML = '<p style="color:red;">Invalid image file selected.</p>';
    }
  });
});
