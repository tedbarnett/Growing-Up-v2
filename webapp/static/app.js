/* Growing Up - Dashboard JS */

// -----------------------------------------------------------------------
// Filesystem Browse Modal (Finder-style)
// -----------------------------------------------------------------------

var _browseState = {
    targetInputId: null,
    browseType: 'dirs',
    extFilter: '',
    currentPath: '',
    selectedPath: '',
};

function openBrowseModal(inputId, browseType, extFilter) {
    _browseState.targetInputId = inputId;
    _browseState.targetElement = null;
    _browseState.browseType = browseType || 'dirs';
    _browseState.extFilter = extFilter || '';
    _browseState.selectedPath = '';

    var title = browseType === 'files' ? 'Select a File' : 'Select a Folder';
    document.getElementById('browse-modal-title').textContent = title;

    var selectBtn = document.getElementById('browse-select-btn');
    selectBtn.textContent = browseType === 'files' ? 'Select File' : 'Select Folder';

    document.getElementById('browse-modal').classList.remove('hidden');

    populateSidebar();

    var currentVal = document.getElementById(inputId).value;
    var startPath = '';
    if (currentVal && currentVal.startsWith('/')) {
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

function populateSidebar() {
    var sidebar = document.getElementById('browse-sidebar');
    if (!sidebar) return;

    var home = window.HOME_DIR || '';
    if (!home) return;

    var locations = [
        { name: 'Home', path: home, icon: '\uD83C\uDFE0' },
        { name: 'Desktop', path: home + '/Desktop', icon: '\uD83D\uDDA5' },
        { name: 'Documents', path: home + '/Documents', icon: '\uD83D\uDCC4' },
        { name: 'Pictures', path: home + '/Pictures', icon: '\uD83D\uDDBC' },
        { name: 'Downloads', path: home + '/Downloads', icon: '\uD83D\uDCE5' },
    ];

    sidebar.innerHTML = '<div class="browse-sidebar-heading">Favorites</div>';
    locations.forEach(function(loc) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'browse-sidebar-btn';
        btn.setAttribute('data-path', loc.path);
        btn.innerHTML = '<span class="sidebar-icon">' + loc.icon + '</span> ' + loc.name;
        btn.onclick = function() { browseNavigate(loc.path); };
        sidebar.appendChild(btn);
    });
}

function renderBreadcrumbs(path) {
    var container = document.getElementById('browse-breadcrumbs');
    if (!container) return;
    container.innerHTML = '';

    if (!path || path === '/') {
        var span = document.createElement('span');
        span.className = 'browse-crumb browse-crumb-current';
        span.textContent = '/';
        container.appendChild(span);
        return;
    }

    var parts = path.split('/').filter(function(p) { return p !== ''; });

    var rootBtn = document.createElement('button');
    rootBtn.type = 'button';
    rootBtn.className = 'browse-crumb browse-crumb-btn';
    rootBtn.textContent = '/';
    rootBtn.onclick = function() { browseNavigate('/'); };
    container.appendChild(rootBtn);

    parts.forEach(function(part, i) {
        var sep = document.createElement('span');
        sep.className = 'browse-crumb-sep';
        sep.textContent = '\u203A';
        container.appendChild(sep);

        var partPath = '/' + parts.slice(0, i + 1).join('/');

        if (i === parts.length - 1) {
            var span = document.createElement('span');
            span.className = 'browse-crumb browse-crumb-current';
            span.textContent = part;
            container.appendChild(span);
        } else {
            var btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'browse-crumb browse-crumb-btn';
            btn.textContent = part;
            (function(p) {
                btn.onclick = function() { browseNavigate(p); };
            })(partPath);
            container.appendChild(btn);
        }
    });
}

function updateSidebarActive(currentPath) {
    var sidebar = document.getElementById('browse-sidebar');
    if (!sidebar) return;
    var btns = sidebar.querySelectorAll('.browse-sidebar-btn');
    var bestMatch = -1;
    var bestLen = 0;
    btns.forEach(function(btn, i) {
        var loc = btn.getAttribute('data-path');
        if (loc && (currentPath === loc || currentPath.startsWith(loc + '/'))) {
            if (loc.length > bestLen) {
                bestMatch = i;
                bestLen = loc.length;
            }
        }
    });
    btns.forEach(function(btn, i) {
        btn.classList.toggle('active', i === bestMatch);
    });
}

function browseNavigate(path) {
    var params = new URLSearchParams();
    if (path) params.set('path', path);

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

            renderBreadcrumbs(data.current);

            var container = document.getElementById('browse-items');
            container.innerHTML = '';

            if (data.error) {
                container.innerHTML = '<div class="browse-error">' + data.error + '</div>';
                return;
            }

            if (_browseState.browseType === 'files') {
                var dirs = [];
                var files = [];
                data.items.forEach(function(item) {
                    if (item.is_dir) dirs.push(item);
                    else files.push(item);
                });

                dirs.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-dir';
                    el.innerHTML = '<span class="browse-item-icon">\uD83D\uDCC1</span> ' + escapeHtml(item.name);
                    el.onclick = function() { browseNavigate(item.path); };
                    container.appendChild(el);
                });

                files.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-file';
                    if (_browseState.selectedPath === item.path) {
                        el.className += ' browse-item-selected';
                    }
                    el.innerHTML = '<span class="browse-item-icon">\uD83C\uDFB5</span> ' + escapeHtml(item.name);
                    el.onclick = function() {
                        _browseState.selectedPath = item.path;
                        document.getElementById('browse-selected-label').textContent = item.name;
                        container.querySelectorAll('.browse-item-file').forEach(function(f) {
                            f.classList.remove('browse-item-selected');
                        });
                        el.classList.add('browse-item-selected');
                    };
                    container.appendChild(el);
                });
            } else {
                data.items.forEach(function(item) {
                    var el = document.createElement('div');
                    el.className = 'browse-item browse-item-dir';
                    el.innerHTML = '<span class="browse-item-icon">\uD83D\uDCC1</span> ' + escapeHtml(item.name);
                    el.onclick = function() { browseNavigate(item.path); };
                    container.appendChild(el);
                });
            }

            if (data.items.length === 0) {
                container.innerHTML = '<div class="browse-empty">Empty folder</div>';
            }

            if (_browseState.browseType === 'dirs') {
                var pathParts = data.current.split('/');
                var folderName = pathParts[pathParts.length - 1] || '/';
                document.getElementById('browse-selected-label').textContent = folderName;
            }

            updateSidebarActive(data.current);
        });
}

function escapeHtml(text) {
    var el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

function browseConfirm() {
    var value = '';
    if (_browseState.browseType === 'files') {
        value = _browseState.selectedPath;
    } else {
        value = _browseState.currentPath;
    }

    if (value) {
        var input = _browseState.targetElement ||
                    (_browseState.targetInputId ? document.getElementById(_browseState.targetInputId) : null);
        if (input) {
            input.value = value;
            // Trigger change event for path display + settings tracking
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
    closeBrowseModal();
}


// -----------------------------------------------------------------------
// Path Basename Display
// -----------------------------------------------------------------------

function getBasename(path) {
    if (!path) return '';
    var parts = path.replace(/\/+$/, '').split('/');
    return parts[parts.length - 1] || path;
}

function updatePathDisplay(input) {
    var display = input._pathDisplay;
    if (!display) return;
    var val = input.value.trim();
    if (!val) {
        display.style.display = 'none';
        return;
    }
    // Show basename text; add custom tooltip span for full path
    display.style.display = '';
    var nameSpan = display.querySelector('.path-basename-text');
    var tipSpan = display.querySelector('.path-tooltip');
    if (!nameSpan) {
        display.innerHTML = '';
        nameSpan = document.createElement('span');
        nameSpan.className = 'path-basename-text';
        tipSpan = document.createElement('span');
        tipSpan.className = 'path-tooltip';
        display.appendChild(nameSpan);
        display.appendChild(tipSpan);
    }
    nameSpan.textContent = getBasename(val);
    tipSpan.textContent = val;
}

function initPathDisplayForWrapper(wrapper) {
    var input = wrapper.querySelector('input');
    if (!input || input._pathDisplay) return;

    var display = document.createElement('div');
    display.className = 'path-basename';
    wrapper.parentNode.insertBefore(display, wrapper.nextSibling);
    input._pathDisplay = display;

    input.addEventListener('input', function() { updatePathDisplay(input); });
    input.addEventListener('change', function() { updatePathDisplay(input); });
    input.addEventListener('focus', function() {
        display.classList.add('show-tooltip');
        input.removeAttribute('title');
    });
    input.addEventListener('blur', function() {
        setTimeout(function() { display.classList.remove('show-tooltip'); }, 150);
    });
    if (input.hasAttribute('list')) {
        var datalistId = input.getAttribute('list');
        var datalist = document.getElementById(datalistId);
        if (datalist) datalist.remove();
        input.removeAttribute('list');
    }
    updatePathDisplay(input);
}

function initPathDisplays() {
    document.querySelectorAll('.input-with-browse').forEach(function(wrapper) {
        initPathDisplayForWrapper(wrapper);
    });
}

document.addEventListener('DOMContentLoaded', initPathDisplays);


// -----------------------------------------------------------------------
// Multi-Music Row Management
// -----------------------------------------------------------------------

function openBrowseForMusic(btn) {
    var row = btn.closest('.music-row');
    var input = row.querySelector('input[name="music"]');
    _browseState.targetInputId = null;
    _browseState.targetElement = input;
    _browseState.browseType = 'files';
    _browseState.extFilter = '.mp3';
    _browseState.selectedPath = '';

    document.getElementById('browse-modal-title').textContent = 'Select a File';
    document.getElementById('browse-select-btn').textContent = 'Select File';
    document.getElementById('browse-modal').classList.remove('hidden');

    populateSidebar();

    var currentVal = input.value;
    var startPath = '';
    if (currentVal && currentVal.startsWith('/')) {
        var lastSlash = currentVal.lastIndexOf('/');
        startPath = lastSlash > 0 ? currentVal.substring(0, lastSlash) : '/';
    }
    browseNavigate(startPath);
}

function addMusicRow() {
    var list = document.getElementById('music-list');
    if (!list) return;
    var row = document.createElement('div');
    row.className = 'music-row';
    row.innerHTML =
        '<div class="input-with-browse">' +
        '<input type="text" name="music" value="" placeholder="None (or browse for .mp3)...">' +
        '<button type="button" class="btn btn-secondary btn-browse" onclick="openBrowseForMusic(this)">Browse</button>' +
        '<button type="button" class="btn btn-danger btn-sm music-remove-btn" onclick="removeMusicRow(this)">\u00d7</button>' +
        '</div>';
    list.appendChild(row);
    updateMusicRemoveButtons();
    initPathDisplayForWrapper(row.querySelector('.input-with-browse'));
    if (window._checkForChanges) window._checkForChanges();
}

function removeMusicRow(btn) {
    var row = btn.closest('.music-row');
    if (row) row.remove();
    updateMusicRemoveButtons();
    if (window._checkForChanges) window._checkForChanges();
}

function updateMusicRemoveButtons() {
    var rows = document.querySelectorAll('#music-list .music-row');
    var removeBtns = document.querySelectorAll('#music-list .music-remove-btn');
    removeBtns.forEach(function(btn) {
        btn.style.display = rows.length > 1 ? '' : 'none';
    });
}


// -----------------------------------------------------------------------
// Settings Form — disable Save until changes
// -----------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function() {
    var form = document.querySelector('.settings-form');
    if (!form) return;

    var saveBtn = form.querySelector('button[type="submit"]');
    if (!saveBtn) return;

    // Capture initial values (non-music fields)
    var initialValues = {};
    form.querySelectorAll('input:not([name="music"]), select').forEach(function(field) {
        if (field.type === 'checkbox') {
            initialValues[field.name] = field.checked;
        } else {
            initialValues[field.name] = field.value;
        }
    });

    // Capture initial music values
    var initialMusicValues = [];
    form.querySelectorAll('input[name="music"]').forEach(function(input) {
        initialMusicValues.push(input.value);
    });

    function checkForChanges() {
        var hasChanges = false;
        form.querySelectorAll('input:not([name="music"]), select').forEach(function(field) {
            if (!field.name) return;
            if (field.type === 'checkbox') {
                if (field.checked !== initialValues[field.name]) hasChanges = true;
            } else {
                if (field.value !== initialValues[field.name]) hasChanges = true;
            }
        });
        // Check music fields (count and values may have changed)
        var currentMusic = [];
        form.querySelectorAll('input[name="music"]').forEach(function(input) {
            currentMusic.push(input.value);
        });
        if (JSON.stringify(currentMusic) !== JSON.stringify(initialMusicValues)) hasChanges = true;

        saveBtn.disabled = !hasChanges;
        if (hasChanges) {
            saveBtn.classList.remove('btn-secondary');
            saveBtn.classList.add('btn-primary');
        } else {
            saveBtn.classList.remove('btn-primary');
            saveBtn.classList.add('btn-secondary');
        }
    }

    // Start disabled
    saveBtn.disabled = true;

    // Use event delegation on the form to catch dynamic inputs
    form.addEventListener('input', checkForChanges);
    form.addEventListener('change', checkForChanges);

    // Expose for use by addMusicRow/removeMusicRow
    window._checkForChanges = checkForChanges;
});


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
                    generateBtn.textContent = window.HAS_VIDEO ? 'Re-Generate Video' : 'Generate Video';
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

        // Update aligned/processed count live
        if (data.aligned_count !== undefined) {
            var stats = document.querySelectorAll('.stat-value');
            if (stats.length >= 2) {
                stats[1].textContent = data.aligned_count;
            }
        }

        if (status.state === 'complete') {
            source.close();
            var phase = status.phase || 'full';

            if (phase === 'process') {
                if (processBtn) {
                    processBtn.textContent = 'Re-Process Images';
                    processBtn.disabled = false;
                }
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.textContent = data.has_video ? 'Re-Generate Video' : 'Generate Video';
                }
                var scrubberSection = document.getElementById('scrubber-section');
                if (scrubberSection) scrubberSection.classList.remove('hidden');
                loadScrubber();
            } else if (phase === 'generate' || phase === 'full') {
                if (processBtn) {
                    processBtn.textContent = 'Re-Process Images';
                    processBtn.disabled = false;
                }
                if (generateBtn) {
                    generateBtn.textContent = 'Re-Generate Video';
                    generateBtn.disabled = false;
                }
                window.HAS_VIDEO = true;
                if (data.has_video) {
                    showVideoPlayer();
                }
            }
        } else if (status.state === 'error') {
            source.close();
            if (processBtn) {
                processBtn.textContent = window.IS_PROCESSED ? 'Re-Process Images' : 'Process Images';
                processBtn.disabled = false;
            }
            if (generateBtn) {
                generateBtn.textContent = (data.has_video || window.HAS_VIDEO) ? 'Re-Generate Video' : 'Generate Video';
                generateBtn.disabled = !data.is_processed;
            }
        }
    };

    source.onerror = function () {
        source.close();
        if (processBtn) {
            processBtn.textContent = window.IS_PROCESSED ? 'Re-Process Images' : 'Process Images';
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
            'download="' + subject + ' - Growing Up.mp4">' +
            '\u2B07 Save Video</a>';
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

function updateScrubberDuration() {
    var el = document.getElementById('scrubber-duration');
    if (!el) return;
    var n = _scrubberState.images.length;
    if (n < 2) { el.textContent = ''; return; }
    var fps = window.PIPELINE_FPS || 30;
    var hold = window.PIPELINE_HOLD || 15;
    var morph = window.PIPELINE_MORPH || 30;
    var titleFrames = 3 * fps;
    var imageFrames = n * hold + hold + (n - 1) * morph;
    var endingFrames = 3 * fps + 3 * fps;
    var totalSeconds = Math.round((titleFrames + imageFrames + endingFrames) / fps);
    var mins = Math.floor(totalSeconds / 60);
    var secs = totalSeconds % 60;
    el.textContent = '(' + mins + ':' + (secs < 10 ? '0' : '') + secs + ' duration)';
}

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
            updateScrubberDuration();
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
                    delete _scrubberState.preloadCache[img.filename];
                    images.splice(_scrubberState.currentIndex, 1);

                    if (images.length === 0) {
                        document.getElementById('scrubber-section').classList.add('hidden');
                        return;
                    }

                    var slider = document.getElementById('scrubber-slider');
                    if (slider) slider.max = images.length - 1;

                    var newIndex = Math.min(_scrubberState.currentIndex, images.length - 1);
                    showScrubberImage(newIndex);

                    var stats = document.querySelectorAll('.stat-value');
                    if (stats.length >= 2) {
                        stats[1].textContent = images.length;
                    }

                    updateScrubberDuration();
                })
                .catch(function (err) {
                    alert('Delete failed: ' + err);
                });
        });
    }
});
