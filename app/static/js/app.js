/**
 * RoadScan Pi — Client-side JavaScript
 * ======================================
 * Drag-and-drop uploads, tab switching, video SSE progress,
 * range slider labels. Zero external dependencies.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ---- Tab Switching ----
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.upload-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const panel = document.getElementById(`panel-${target}`);
            if (panel) panel.classList.add('active');
        });
    });

    // ---- Drag & Drop ----
    function setupDropZone(zoneId, inputId, previewContainerId, previewId, removeId, type) {
        const zone = document.getElementById(zoneId);
        const input = document.getElementById(inputId);
        const previewContainer = document.getElementById(previewContainerId);
        const preview = document.getElementById(previewId);
        const removeBtn = document.getElementById(removeId);

        if (!zone || !input) return;

        // Click to browse
        zone.addEventListener('click', (e) => {
            if (e.target.tagName !== 'LABEL') {
                input.click();
            }
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(evt => {
            zone.addEventListener(evt, (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            zone.addEventListener(evt, (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
            });
        });

        zone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                showPreview(files[0]);
            }
        });

        // File selection
        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                showPreview(input.files[0]);
            }
        });

        function showPreview(file) {
            if (!previewContainer || !preview) return;

            if (type === 'image') {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    previewContainer.style.display = 'block';
                    zone.style.display = 'none';
                };
                reader.readAsDataURL(file);
            } else if (type === 'video') {
                const url = URL.createObjectURL(file);
                preview.src = url;
                previewContainer.style.display = 'block';
                zone.style.display = 'none';
            }
        }

        // Remove preview
        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                input.value = '';
                if (preview.src) {
                    if (type === 'video' && preview.src.startsWith('blob:')) {
                        URL.revokeObjectURL(preview.src);
                    }
                    preview.src = '';
                }
                previewContainer.style.display = 'none';
                zone.style.display = 'flex';
            });
        }
    }

    setupDropZone(
        'image-drop-zone', 'image-file',
        'image-preview-container', 'image-preview', 'image-remove',
        'image'
    );

    setupDropZone(
        'video-drop-zone', 'video-file',
        'video-preview-container', 'video-preview', 'video-remove',
        'video'
    );

    // ---- Range Sliders ----
    function setupRangeSlider(sliderId, displayId, formatter) {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);

        if (!slider || !display) return;

        const update = () => {
            display.textContent = formatter ? formatter(slider.value) : slider.value;
        };

        slider.addEventListener('input', update);
        update();
    }

    setupRangeSlider('image-confidence', 'image-conf-value', v => `${Math.round(v * 100)}%`);
    setupRangeSlider('video-confidence', 'video-conf-value', v => `${Math.round(v * 100)}%`);
    setupRangeSlider('frame-skip', 'skip-value', v => v);

    // ---- Image Form: Loading State ----
    const imageForm = document.getElementById('image-form');
    const imageSubmit = document.getElementById('image-submit');

    if (imageForm && imageSubmit) {
        imageForm.addEventListener('submit', () => {
            imageSubmit.disabled = true;
            imageSubmit.innerHTML = '<span class="spinner"></span><span>Analyzing...</span>';
        });
    }

    // ---- Video Form: SSE Upload ----
    const videoForm = document.getElementById('video-form');
    const videoSubmit = document.getElementById('video-submit');
    const progressContainer = document.getElementById('video-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressEta = document.getElementById('progress-eta');
    const progressFrames = document.getElementById('progress-frames');
    const progressDetections = document.getElementById('progress-detections');
    const videoResult = document.getElementById('video-result');
    const videoDownload = document.getElementById('video-download');

    if (videoForm) {
        videoForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fileInput = document.getElementById('video-file');
            if (!fileInput || !fileInput.files.length) {
                alert('Please select a video file first.');
                return;
            }

            // Prepare form data
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const confSlider = document.getElementById('video-confidence');
            const skipSlider = document.getElementById('frame-skip');
            if (confSlider) formData.append('confidence', confSlider.value);
            if (skipSlider) formData.append('frame_skip', skipSlider.value);

            // Update UI state
            videoSubmit.disabled = true;
            videoSubmit.innerHTML = '<span class="spinner"></span><span>Uploading...</span>';

            try {
                const response = await fetch('/detect/video', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`Upload failed: ${response.status}`);
                }

                // Show progress
                if (progressContainer) progressContainer.style.display = 'block';
                videoSubmit.innerHTML = '<span class="spinner"></span><span>Processing...</span>';

                // Read SSE stream
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));

                                if (data.error) {
                                    throw new Error(data.error);
                                }

                                if (data.progress !== undefined) {
                                    if (progressFill) progressFill.style.width = `${data.progress}%`;
                                    if (progressFrames) progressFrames.textContent = `${data.frame} / ${data.total} frames`;
                                    if (progressEta) progressEta.textContent = `ETA: ${data.eta}s`;
                                    if (progressDetections) progressDetections.textContent = `${data.detections} detections`;
                                }

                                if (data.complete) {
                                    // Show download button
                                    if (progressContainer) progressContainer.style.display = 'none';
                                    if (videoResult) videoResult.style.display = 'block';
                                    if (videoDownload) {
                                        videoDownload.href = `/download/${data.result_file}`;
                                    }
                                    videoSubmit.disabled = false;
                                    videoSubmit.innerHTML = '<span class="btn-icon">🎯</span><span>Process Video</span>';
                                }
                            } catch (parseErr) {
                                console.warn('SSE parse error:', parseErr);
                            }
                        }
                    }
                }

            } catch (err) {
                console.error('Video processing error:', err);
                alert(`Error: ${err.message}`);
                videoSubmit.disabled = false;
                videoSubmit.innerHTML = '<span class="btn-icon">🎯</span><span>Process Video</span>';
                if (progressContainer) progressContainer.style.display = 'none';
            }
        });
    }

    // ---- Auto-dismiss alerts ----
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });

});
