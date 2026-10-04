var uploadForm = document.getElementById('uploadForm');
uploadForm.addEventListener('submit', async function (event) {
    event.preventDefault();
    var button = this.querySelector('button');
    var status = this.querySelector('[role="status"]');
    button.disabled = true;
    status.textContent = 'Extracting text…';
    try {
        var response = await fetch(this.action, { method: 'POST', body: new FormData(this) });
        var json = (response.headers.get('content-type') || '').includes('application/json');
        var success = response.ok && (response.redirected || (json && (await response.json()).status === 'success'));
        if (!success) throw new Error('Upload failed');
        status.textContent = 'Text extraction complete. The extracted text is saved in the local report.txt file.';
    } catch (error) {
        status.textContent = 'Text extraction failed. Check the file and try again.';
    } finally { button.disabled = false; }
});
