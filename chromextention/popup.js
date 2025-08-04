const trivia = [
  {
    question: "Who was the first President of the United States?",
    answers: ["George Washington", "Abraham Lincoln", "Thomas Jefferson", "John Adams"],
    correct: 0
  },
  {
    question: "In which year did World War II end?",
    answers: ["1942", "1945", "1939", "1950"],
    correct: 1
  },
  {
    question: "The Great Wall of China was primarily built to protect against which group?",
    answers: ["Romans", "Mongols", "Vikings", "Huns"],
    correct: 1
  },
  {
    question: "Who discovered penicillin?",
    answers: ["Marie Curie", "Alexander Fleming", "Isaac Newton", "Albert Einstein"],
    correct: 1
  },
  {
    question: "What ancient civilization built Machu Picchu?",
    answers: ["Aztec", "Maya", "Inca", "Olmec"],
    correct: 2
  }
];

let currentIndex = -1;

function getRandomQuestion() {
  let idx;
  do {
    idx = Math.floor(Math.random() * trivia.length);
  } while (idx === currentIndex);
  currentIndex = idx;
  return trivia[idx];
}

function showQuestion() {
  const q = getRandomQuestion();
  document.getElementById('question').textContent = q.question;
  const answersDiv = document.getElementById('answers');
  answersDiv.innerHTML = '';
  q.answers.forEach((ans, i) => {
    const btn = document.createElement('button');
    btn.textContent = ans;
    btn.onclick = () => checkAnswer(i, q.correct);
    btn.className = 'answer-btn';
    answersDiv.appendChild(btn);
  });
  document.getElementById('feedback').textContent = '';
}

function checkAnswer(selected, correct) {
  const feedback = document.getElementById('feedback');
  if (selected === correct) {
    feedback.textContent = 'Correct! 🎉';
    feedback.style.color = 'green';
  } else {
    feedback.textContent = 'Oops! Try again.';
    feedback.style.color = 'red';
  }
}

document.getElementById('next-btn').onclick = showQuestion;

document.getElementById('premium-btn').onclick = function() {
  alert('Unlock more trivia packs and exclusive features coming soon!');
};

// Show first question on load
showQuestion();
