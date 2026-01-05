class TextField {
  constructor() {
    this.input = createElement('textarea');
    this.input.position(windowWidth * 0.25, windowHeight * 0.75);
    this.input.size(windowWidth * 0.5, 20); // start small

    // Styles
    this.input.style('font-size', '16px');
    this.input.style('background-color', '#333333');
    this.input.style('color', '#ffffff');
    this.input.style('border-radius', '10px');
    this.input.style('border', '2px solid #333333');
    this.input.style('padding', '10px');
    this.input.style('resize', 'none');
    this.input.attribute('wrap', 'soft');

    // Listen for input to auto-resize
    this.input.input(() => this.autoResize());

    // Handle Enter vs Shift+Enter
    this.input.elt.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // prevent newline
        this.submitText();
      }
    });

  }

  display() {
    this.input.show();
  }

  autoResize() {
    const padding = 20; // padding-top + padding-bottom in px
    this.input.elt.style.height = '20px'; // reset to minimal height
    this.input.elt.style.height = (this.input.elt.scrollHeight - padding) + 'px';
  }

  submitText() {
    console.log(this.input.value()); // print text
    this.input.value(''); // clear input after submit
    this.autoResize(); // reset height
  }
}