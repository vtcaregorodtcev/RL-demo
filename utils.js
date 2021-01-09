function distance(x1, x2, y1, y2) {
  return Math.sqrt(Math.pow(Math.abs(x1 - x2), 2) + Math.pow(Math.abs(y1 - y2), 2));
}

function toRect(rectable) {
  rectable.bottom = rectable.top + rectable.height;
  rectable.right = rectable.left + rectable.width;

  return rectable;
}

function rectsIntersected(r1, r2) {
  return !(r2.left > r1.right ||
    r2.right < r1.left ||
    r2.top > r1.bottom ||
    r2.bottom < r1.top);
}
