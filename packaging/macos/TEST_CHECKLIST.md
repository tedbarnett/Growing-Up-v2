# Growing Up — macOS App Test Checklist

Run through this before distributing a new build.

## Installation
- [ ] Double-click DMG, drag app to Applications
- [ ] Launch app from Applications (first time — expect longer startup)
- [ ] App opens browser to localhost:5001
- [ ] Launch app again while already running — should open browser, not crash

## Subject Setup
- [ ] Create a new subject from the dashboard
- [ ] Set subject name and birthdate
- [ ] Browse and select an images folder (with 5-10 test photos)
- [ ] Save settings

## Phase 1 — Process Images
- [ ] Click "Process Images"
- [ ] Progress bar updates via SSE
- [ ] No port-in-use or re-launch errors
- [ ] Completes without errors
- [ ] Aligned faces appear in scrubber

## Review
- [ ] Scrubber slider works (arrow keys, dragging)
- [ ] Delete a face, confirm it's removed
- [ ] Age labels display correctly
- [ ] Duration estimate updates after deletion

## Phase 2 — Generate Video
- [ ] Add an MP3 file for music (optional)
- [ ] Click "Generate Video"
- [ ] Progress bar updates
- [ ] Completes without errors
- [ ] Video plays in browser

## General
- [ ] Dashboard shows correct badges (Processed, Video Generated, etc.)
- [ ] Multiple subjects work independently
- [ ] App quits cleanly (Cmd+Q or close terminal)
- [ ] No leftover processes after quit (`lsof -i :5001`)
