document.querySelectorAll('[data-busy-form]').forEach(function (form) {
    form.addEventListener('submit', function () {
        var button = form.querySelector('button[type="submit"]');
        button.disabled = true;
        button.textContent = button.dataset.busyLabel;
        form.setAttribute('aria-busy', 'true');
        var rebuild = form.querySelector('#index-rebuild:checked');
        form.querySelector('[role="status"]').textContent = rebuild
            ? 'Refreshing sources and analyzing. This may take several minutes.'
            : 'Preparing your response. Please keep this page open.';
    });
});
window.addEventListener('pageshow', function () {
    document.querySelectorAll('[data-busy-form]').forEach(function (form) {
        var button = form.querySelector('button[type="submit"]');
        button.disabled = false;
        button.textContent = 'Analyze →';
        form.removeAttribute('aria-busy');
        form.querySelector('[role="status"]').textContent = '';
    });
});
