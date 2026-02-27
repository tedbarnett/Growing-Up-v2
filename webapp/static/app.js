/* Growing Up - Dashboard JS */

// -----------------------------------------------------------------------
// Filesystem Browse Modal
// -----------------------------------------------------------------------

var _browseState = {
    targetInputId: null,   // which <input> to populate
    browseType: 'dirs',    // "dirs" or "files"
    extFilter: '',         // e.g. ".mp3"
    currentPath: '',
    selectedPath: '',
};

function openBrowseModal(inputId, browseType, extFilter) {
    _browseState.targetInputId = inputId;
    _browseState.browseType = browseType || 'dirs';
    _browseState.extFilter = extFilter || '';
    _browseState.selectedPath = '';

    var title = browseType === 'files' ? 'Browse for File' : 'Browse for Folder';
    document.getElementById('browse-modal-title').textContent = title;

    var selectBtn = document.getElementById('browse-select-btn');
    selectBtn.textContent = browseType === 'files' ? 'Select File' : 'Select This Folder';

    document.getElementById('browse-modal').classList.remove('hidden');

    // Start from current input value or home
    var currentVal = document.getElementById(inputId).value;
    var startPath = '';
    if (currentVal && currentVal.startsWith('/')) {
        // For files, start in the parent directory
        if (browseType === 'files') {
            var lastSlash = currentVal.lastIndexOf('/');
            startPath = lastSlash > 0 ? currentVal.substring(0, lastSlash) : '/';
        } else {
            startPath = currentVal;
        }
    }
    browseNavigate(startPath);
}

function closeBrowseModal() {
    document.getElementById('browse-modal').classList.add('hidden');
}

function browseNavigate(path) {
    var params = new URLSearchParams();
    if (path) params.set('path', path);

    // For file browsing, we always list dirs to navigate, plus files to select
    if (_browseState.browseType === 'files') {
        params.set('type', 'all');
        if (_browseState.extFilter) params.set('ext', _browseState.extFilter);
    } else {
        params.set('type', 'dirs');
    }

    fetch('/api/browse?' + params.toString())
        .then(function(r) { return r.json(); })
        .then(function(data) {
            _browseState.currentPath = data.current;
            document.getElementById('browse-path-input').value = data.current;

            var container = document.getElementById('browse-items');
            container.innerHTML = '';

            if (data.error) {
                container.innerHTML = '<div class="browse-error">' + data.error + '</div>';
                return;
            }

            // For file browsing, show dirs and matching files
            if (_browseState.browseType === 'files') {
                // Show directories first (for navigation)
                var dirs = [];
                var files = [];
                data.items.forEach(function(item) {
                    if (item.is_dir) dirs.push(item);
                    else files.push(item);
                });

                dirs.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-dir';
                    el.textContent = item.name + '/';
                    el.onclick = function() { browseNavigate(item.path); };
                    container.appendChild(el);
                });

                files.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-file';
                    if (_browseState.selectedPath === item.path) {
                        el.className += ' browse-item-selected';
                    }
                    el.textContent = item.name;
                    el.onclick = function() {
                        _browseState.selectedPath = item.path;
                        document.getElementById('browse-selected-label').textContent = item.name;
                        // Update selection highlighting
                        container.querySelectorAll('.browse-item-file').forEach(function(f) {
                            f.classList.remove('browse-item-selected');
                        });
                        el.classList.add('browse-item-selected');
                    };
                    container.appendChild(el);
                });
            } else {
                // Directory browsing — show subdirs to navigate into
                data.items.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-dir';
                    el.textContent = item.name + '/';
                    el.onclick = function() { browseNavigate(item.path); };
                    container.appendChild(el);
                });
            }

            if (data.items.length === 0) {
                container.innerHTML = '<div class="browse-empty">No items</div>';
            }

            // Update selected label for directory mode
            if (_browseState.browseType === 'dirs') {
                document.getElementById('browse-selected-label').textContent = data.current;
            }
        });
}

function browseUp() {
    var pathInput = document.getElementById('browse-path-input');
    var current = pathInput.value || _browseState.currentPath;
    // Go to parent
    var lastSlash = current.lastIndexOf('/');
    var parent = lastSlash > 0 ? current.substring(0, lastSlash) : '/';
    browseNavigate(parent);
}

function browseConfirm() {
    var value = '';
    if (_browseState.browseType === 'files') {
        value = _browseState.selectedPath;
    } else {
        value = _browseState.currentPath;
    }

    if (value && _browseState.targetInputId) {
        document.getElementById(_browseState.targetInputId).value = value;
    }
    closeBrowseModal();
}


// -----------------------------------------------------------------------
// Delete Subject
// -----------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function () {
    var deleteBtn = document.getElementById('delete-subject-btn');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', function () {
            var subject = deleteBtn.dataset.subject;
            if (!confirm('Delete subject "' + subject + '"?\n\nThis removes all processed data (aligned faces, frames, video). Original images are NOT deleted.')) return;

            fetch('/subjects/' + encodeURIComponent(subject) + '/delete', { method: 'DELETE' })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    window.location.href = '/';
                })
                .catch(function (err) { alert('Delete failed: ' + err); });
        });
    }
});


// -----------------------------------------------------------------------
// Process Images + Generate Video button handlers
// -----------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function () {
    var processBtn = document.getElementById('process-btn');
    var generateBtn = document.getElementById('generate-btn');

    if (processBtn) {
        processBtn.addEventListener('click', function () {
            var subject = processBtn.dataset.subject;
            processBtn.disabled = true;
            processBtn.textContent = 'Processing...';
            if (generateBtn) generateBtn.disabled = true;

            fetch('/subjects/' + encodeURIComponent(subject) + '/process-images', {
                method: 'POST',
            })
                .then(function (resp) { return resp.json(); })
                .then(function (data) {
                    if (data.error) {
                        processBtn.textContent = 'Error: ' + data.error;
                        processBtn.disabled = false;
                        return;
                    }
                    startStatusStream();
                })
                .catch(function (err) {
                    processBtn.textContent = 'Process Images';
                    processBtn.disabled = false;
                    console.error(err);
                });
        });
    }

    if (generateBtn) {
        generateBtn.addEventListener('click', function () {
            var subject = generateBtn.dataset.subject;
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            if (processBtn) processBtn.disabled = true;

            fetch('/subjects/' + encodeURIComponent(subject) + '/generate-video', {
                method: 'POST',
            })
                .then(function (resp) { return resp.json(); })
                .then(function (data) {
                    if (data.error) {
                        generateBtn.textContent = 'Error: ' + data.error;
                        generateBtn.disabled = false;
                        return;
                    }
                    startStatusStream();
                })
                .catch(function (err) {
                    generateBtn.textContent = 'Generate Video';
                    generateBtn.disabled = false;
                    console.error(err);
                });
        });
    }
});


// -----------------------------------------------------------------------
// SSE Status Stream (phase-aware)
// -----------------------------------------------------------------------

function startStatusStream() {
    var subject = window.SUBJECT_NAME;
    if (!subject) return;

    var progressSection = document.getElementById('progress-section');
    var stepLabel = document.getElementById('progress-step-label');
    var stepCount = document.getElementById('progress-step-count');
    var progressBar = document.getElementById('progress-bar');
    var logOutput = document.getElementById('log-output');
    var processBtn = document.getElementById('process-btn');
    var generateBtn = document.getElementById('generate-btn');

    if (progressSection) progressSection.classList.remove('hidden');

    var source = new EventSource('/subjects/' + encodeURIComponent(subject) + '/status');

    source.onmessage = function (event) {
        var data = JSON.parse(event.data);
        var status = data.status;

        if (!status) return;

        if (stepLabel) stepLabel.textContent = status.step_label || '';
        if (stepCount && status.total_steps) {
            stepCount.textContent = 'Step ' + (status.step_index + 1) + '/' + status.total_steps;
        }
        if (progressBar && status.total_steps) {
            var pct = ((status.step_index + 1) / status.total_steps) * 100;
            progressBar.style.width = pct + '%';
        }
        if (logOutput && status.log_tail) {
            logOutput.textContent = status.log_tail.join('\n');
            logOutput.scrollTop = logOutput.scrollHeight;
        }

        if (status.state === 'complete') {
            source.close();
            var phase = status.phase || 'full';

            if (phase === 'process') {
                // Phase 1 done — show scrubber, enable generate
                if (processBtn) {
                    processBtn.textContent = 'Process Images';
                    processBtn.disabled = false;
                }
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.textContent = 'Generate Video';
                }
                // Show scrubber
                var scrubberSection = document.getElementById('scrubber-section');
                if (scrubberSection) scrubberSection.classList.remove('hidden');
                loadScrubber();
            } else if (phase === 'generate' || phase === 'full') {
                // Phase 2 done — show video
                if (processBtn) {
                    processBtn.textContent = 'Process Images';
                    processBtn.disabled = false;
                }
                if (generateBtn) {
                    generateBtn.textContent = 'Generate Video';
                    generateBtn.disabled = false;
                }
                if (data.has_video) {
                    showVideoPlayer();
                }
            }
        } else if (status.state === 'error') {
            source.close();
            if (processBtn) {
                processBtn.textContent = 'Process Images';
                processBtn.disabled = false;
            }
            if (generateBtn) {
                generateBtn.textContent = 'Generate Video';
                generateBtn.disabled = data.is_processed ? false : true;
            }
        }
    };

    source.onerror = function () {
        source.close();
        if (processBtn) {
            processBtn.textContent = 'Process Images';
            processBtn.disabled = false;
        }
        if (generateBtn) {
            generateBtn.textContent = 'Generate Video';
            generateBtn.disabled = false;
        }
    };
}


// -----------------------------------------------------------------------
// Video Player
// -----------------------------------------------------------------------

function showVideoPlayer() {
    var subject = window.SUBJECT_NAME;
    var container = document.getElementById('video-container');
    if (container) {
        container.innerHTML =
            '<h2>Video</h2>' +
            '<video id="video-player" controls preload="metadata">' +
            '<source src="/subjects/' + encodeURIComponent(subject) + '/video" type="video/mp4">' +
            '</video>' +
            '<a id="save-video-btn" class="btn btn-secondary btn-large" ' +
            'href="/subjects/' + encodeURIComponent(subject) + '/video" ' +
            'download="Growing Up - ' + subject + '.mp4">' +
            'Save Video</a>';
    }
}


// -----------------------------------------------------------------------
// Age Label Computation
// -----------------------------------------------------------------------

function computeAgeLabel(birthdateStr, photoDateStr) {
    if (!birthdateStr || !photoDateStr) return '';
    var birthParts = birthdateStr.split('-');
    var photoParts = photoDateStr.split('-');
    if (birthParts.length < 2 || photoParts.length < 1) return '';

    var birthYear = parseInt(birthParts[0], 10);
    var birthMonth = parseInt(birthParts[1], 10);
    var birthDay = birthParts.length >= 3 ? parseInt(birthParts[2], 10) : 1;

    var photoYear = parseInt(photoParts[0], 10);
    var photoMonth = photoParts.length >= 2 ? parseInt(photoParts[1], 10) : 6;
    var photoDay = photoParts.length >= 3 ? parseInt(photoParts[2], 10) : 15;

    if (isNaN(birthYear) || isNaN(photoYear)) return '';
    if (photoYear < birthYear) return '';

    var months = (photoYear - birthYear) * 12 + (photoMonth - birthMonth);
    if (photoDay < birthDay) months -= 1;

    if (months < 1) return 'Newborn';
    if (months === 1) return '1 month';
    if (months < 12) return months + ' months';
    var years = Math.floor(months / 12);
    return 'Age ' + years;
}


// -----------------------------------------------------------------------
// Scrubber / Flipbook
// -----------------------------------------------------------------------

var _scrubberState = {
    images: [],
    currentIndex: 0,
    preloadCache: {},
};

function loadScrubber() {
    var subject = window.SUBJECT_NAME;
    if (!subject) return;

    fetch('/subjects/' + encodeURIComponent(subject) + '/aligned-sequence')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            _scrubberState.images = data.images || [];
            _scrubberState.currentIndex = 0;

            if (_scrubberState.images.length === 0) {
                var scrubberSection = document.getElementById('scrubber-section');
                if (scrubberSection) scrubberSection.classList.add('hidden');
                return;
            }

            var slider = document.getElementById('scrubber-slider');
            if (slider) {
                slider.max = _scrubberState.images.length - 1;
                slider.value = 0;
            }

            showScrubberImage(0);
            preloadNearby(0);
        });
}

function showScrubberImage(index) {
    var images = _scrubberState.images;
    if (index < 0 || index >= images.length) return;

    _scrubberState.currentIndex = index;
    var img = images[index];
    var subject = window.SUBJECT_NAME;

    var imgEl = document.getElementById('scrubber-image');
    if (imgEl) {
        imgEl.src = '/subjects/' + encodeURIComponent(subject) + '/aligned/' + encodeURIComponent(img.filename);
    }

    var yearLabel = document.getElementById('scrubber-year-label');
    if (yearLabel) {
        var ageText = computeAgeLabel(window.SUBJECT_BIRTHDATE, img.sort_date || img.date);
        yearLabel.textContent = ageText || img.year || '';
    }

    var counter = document.getElementById('scrubber-counter');
    if (counter) {
        counter.textContent = (index + 1) + ' / ' + images.length;
    }

    var filenameEl = document.getElementById('scrubber-filename');
    if (filenameEl) {
        filenameEl.textContent = img.filename;
    }

    var slider = document.getElementById('scrubber-slider');
    if (slider) {
        slider.value = index;
    }

    preloadNearby(index);
}

function preloadNearby(index) {
    var images = _scrubberState.images;
    var subject = window.SUBJECT_NAME;
    for (var i = Math.max(0, index - 5); i <= Math.min(images.length - 1, index + 5); i++) {
        var fname = images[i].filename;
        if (!_scrubberState.preloadCache[fname]) {
            var preImg = new Image();
            preImg.src = '/subjects/' + encodeURIComponent(subject) + '/aligned/' + encodeURIComponent(fname);
            _scrubberState.preloadCache[fname] = preImg;
        }
    }
}

// Scrubber controls
document.addEventListener('DOMContentLoaded', function () {
    var slider = document.getElementById('scrubber-slider');
    var prevBtn = document.getElementById('scrubber-prev');
    var nextBtn = document.getElementById('scrubber-next');
    var deleteBtn = document.getElementById('scrubber-delete-btn');

    if (slider) {
        slider.addEventListener('input', function () {
            showScrubberImage(parseInt(slider.value, 10));
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', function () {
            if (_scrubberState.currentIndex > 0) {
                showScrubberImage(_scrubberState.currentIndex - 1);
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', function () {
            if (_scrubberState.currentIndex < _scrubberState.images.length - 1) {
                showScrubberImage(_scrubberState.currentIndex + 1);
            }
        });
    }

    // Arrow key support
    document.addEventListener('keydown', function (e) {
        var scrubberSection = document.getElementById('scrubber-section');
        if (!scrubberSection || scrubberSection.classList.contains('hidden')) return;
        // Don't capture arrows when typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            if (_scrubberState.currentIndex > 0) {
                showScrubberImage(_scrubberState.currentIndex - 1);
            }
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            if (_scrubberState.currentIndex < _scrubberState.images.length - 1) {
                showScrubberImage(_scrubberState.currentIndex + 1);
            }
        }
    });

    // Delete button
    if (deleteBtn) {
        deleteBtn.addEventListener('click', function () {
            var images = _scrubberState.images;
            if (images.length === 0) return;

            var img = images[_scrubberState.currentIndex];
            if (!confirm('Remove "' + img.filename + '" from the set?')) return;

            var subject = window.SUBJECT_NAME;
            fetch('/subjects/' + encodeURIComponent(subject) + '/aligned/' + encodeURIComponent(img.filename), {
                method: 'DELETE',
            })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    // Remove from scrubber state
                    delete _scrubberState.preloadCache[img.filename];
                    images.splice(_scrubberState.currentIndex, 1);

                    if (images.length === 0) {
                        document.getElementById('scrubber-section').classList.add('hidden');
                        return;
                    }

                    // Update slider max
                    var slider = document.getElementById('scrubber-slider');
                    if (slider) slider.max = images.length - 1;

                    // Show next (or last) image
                    var newIndex = Math.min(_scrubberState.currentIndex, images.length - 1);
                    showScrubberImage(newIndex);

                    // Update aligned count in stats
                    var alignedStat = document.querySelector('.stat-value');
                    if (alignedStat) {
                        // Find the second stat (aligned faces)
                        var stats = document.querySelectorAll('.stat-value');
                        if (stats.length >= 2) {
                            stats[1].textContent = images.length;
                        }
                    }
                })
                .catch(function (err) {
                    alert('Delete failed: ' + err);
                });
        });
    }
});
