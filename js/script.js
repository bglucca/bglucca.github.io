if (document.getElementById('my-work-link')) {
  document.getElementById('my-work-link').addEventListener('click', () => {
    document.getElementById('my-work-section').scrollIntoView({behavior: "smooth"})
  })
}

if (document.getElementById('experience-section-link')) {
  document.getElementById('experience-section-link').addEventListener('click', () => {
    document.getElementById('experience-section').scrollIntoView({behavior: "smooth"})
  })
}