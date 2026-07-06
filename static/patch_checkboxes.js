/**
 * Gradio Checkbox Group styling patch.
 * Targets Gradio 6's rendering quirks to support rounded pill styles
 * without native checkbox squares showing.
 */
function patchCheckboxes() {
    const checks = document.querySelectorAll('.checkbox-group .check');
    checks.forEach(el => {
        el.style.setProperty('display', 'none', 'important');
        el.style.setProperty('width', '0', 'important');
        el.style.setProperty('height', '0', 'important');
    });

    const inputs = document.querySelectorAll('.checkbox-group input[type="checkbox"]');
    inputs.forEach(el => {
        el.style.setProperty('display', 'none', 'important');
    });

    // Support hiding checkmark SVGs inside labels
    const svgs = document.querySelectorAll('.checkbox-group label svg');
    svgs.forEach(el => {
        el.style.setProperty('display', 'none', 'important');
    });
}

// Wire up events
document.addEventListener('DOMContentLoaded', patchCheckboxes);
new MutationObserver(patchCheckboxes).observe(document.documentElement, {
    childList: true, 
    subtree: true
});
setInterval(patchCheckboxes, 300);
