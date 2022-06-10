const NODES = 3;

let matrixATextBoxes = [];
let matrixA = [];

function createInputMatrix() {
  let tableElem = document.getElementById('matrixA');
  let tbodyElem = document.createElement('tbody');
  for (let i = 0; i < NODES; i++) {
    let rowElem = document.createElement('tr');
    for (let j = 0; j < NODES; j++) {
      let tdElem = document.createElement('td');
      let cellElem = document.createElement('input', 'text');
      tdElem.appendChild(cellElem);
      cellElem.value = '0';

      cellElem.onkeydown = function (evt) {
        if (evt.target.value.length >= 4 && evt.keyCode != 8) return false;
        if (evt.target.value.length == 0 && evt.keyCode == 173) return true;
        var ASCIICode = evt.which ? evt.which : evt.keyCode;
        if (ASCIICode > 31 && (ASCIICode < 48 || ASCIICode > 57)) return false;
        return true;
      };

      cellElem.onfocus = function (evt) {
        evt.target.value = '';
      };

      rowElem.appendChild(tdElem);
      matrixATextBoxes.push(cellElem);

      matrixA.push(0);
    }
    tbodyElem.appendChild(rowElem);
  }
  tableElem.appendChild(tbodyElem);
}

function solve() {
  matrixA = [];
  let row = [];
  for (let i = 0; i < matrixATextBoxes.length; i++) {
    row.push(Number.parseInt(matrixATextBoxes[i].value));

    if ((i + 1) % 3 == 0) {
      matrixA.push(row);
      row = [];
    }
  }

  let r = findU(matrixA);
  let mathJaxRes = getMathJax(r[0], r[1], r[2]);
  document.getElementById('resultDiv').innerHTML = mathJaxRes;
  MathJax.typeset();
}

function createIdentityMatrix(size) {
  let I = [];
  for (let i = 0; i < size; i++) {
    let row = [];
    for (let j = 0; j < size; j++) {
      let s = 0;
      if (i == j) s = 1;
      row.push(s);
    }
    I.push(row);
  }
  return I;
}

function createElementaryMatrix(val, x, y) {
  let E = createIdentityMatrix(NODES);
  E[x][y] = val;
  return E;
}

function findU(X) {
  let A = X.map((arr) => arr.slice());
  let E_s = [];
  for (let i = 0; i < NODES; i++) {
    let pivot = -1;
    let divider = -1;
    let k = 1;
    for (let j = i; j < NODES; j++) {
      if (i + k >= NODES) break;
      if (j == i) {
        pivot = A[i][j];
        if (i != NODES - 1 && A[i + k][j] != 0) {
          divider = (-1 * A[i + k][j]) / pivot;
          E_s.push(createElementaryMatrix(divider, i + k, j));
        } else {
          continue;
        }
      }
      A[i + k][j] = divider * A[i][j] + A[i + k][j];
      if (j == NODES - 1 && i + k < NODES) {
        j = i - 1;
        k += 1;
      }
    }
  }
  return [X, A, E_s];
}

function findIndexes(E) {
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      if (E[i][j] != 1 && E[i][j] != 0) {
        return [i + 1, j + 1];
      }
    }
  }
  return [-1, -1];
}

function getElementaryInverse(E) {
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      if (E[i][j] != 1 && E[i][j] != 0) {
        E[i][j] *= -1;
      }
    }
  }
  return E;
}

function formatNumberMathJax(num) {
  if (Number.isInteger(num)) {
    return num;
  } else {
    frac = toFraction(num);
    let n = frac[0];
    let d = frac[1];
    return '\\frac{' + n + '}{' + d + '}';
  }
}

function getMatrixMathJax(M) {
  let mathJaxRes = String.raw`\begin{bmatrix}`;
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      mathJaxRes += formatNumberMathJax(M[i][j]) + (j != NODES - 1 ? '&' : '');
    }
    mathJaxRes += i != NODES - 1 ? String.raw`\\` : '';
  }
  mathJaxRes += String.raw`\end{bmatrix}`;
  return mathJaxRes;
}

function matrixMult(E, A) {
  T = createIdentityMatrix(NODES);
  for (let i = 0; i < NODES; i++) {
    for (let j = 0; j < NODES; j++) {
      s = 0;
      for (let k = 0; k < NODES; k++) {
        s += E[i][k] * A[k][j];
      }
      T[i][j] = s;
    }
  }
  return T;
}

function getMathJax(A, U, E_s) {
  let res = '$$ A = LU $$';
  res += '$$A = ' + getMatrixMathJax(A) + '$$';
  res += "Now let's do Gaussian Elimination on A to make U:";

  let T = A.map((arr) => arr.slice());

  E_stack = [];

  for (let i = 0; i < E_s.length; i++) {
    E = E_s[i];
    indexes = findIndexes(E);
    res +=
      '$$E_{' +
      indexes[0] +
      ',' +
      indexes[1] +
      '} = ' +
      getMatrixMathJax(E) +
      '$$';

    E_stack.push('E_{' + indexes[0] + ',' + indexes[1] + '}');

    res += '$$' + getMatrixMathJax(E) + '.' + getMatrixMathJax(T);

    T = matrixMult(E, T);

    res += ' = ' + getMatrixMathJax(T) + '$$';
  }
  res += '$$U = ' + getMatrixMathJax(U) + '$$';
  res += '$$ L = ';
  while (E_stack.length != 0) {
    res += E_stack.pop() + '^{-1}' + (E_stack.length != 0 ? '.' : '');
  }
  res += '$$';
  res += '$$L = ';
  T = getElementaryInverse(E_s[E_s.length - 1]).map((arr) => arr.slice());
  if (E_s.length >= 2) {
    for (let i = E_s.length - 2; i >= 0; i--) {
      E_I = getElementaryInverse(E_s[i]);
      T = matrixMult(E_I, T);
    }
  }
  res += getMatrixMathJax(T) + '$$';

  res += '$$Solution:$$';
  res +=
    '$$' +
    getMatrixMathJax(A) +
    ' = ' +
    getMatrixMathJax(T) +
    '.' +
    getMatrixMathJax(U) +
    '$$';
  return res;
}

//util function to convert a dec to a frac source: https://stackoverflow.com/questions/23575218/convert-decimal-number-to-fraction-in-javascript-or-closest-fraction
//credit: https://stackoverflow.com/users/2105930/chowey
function toFraction(x, tolerance) {
  if (x == 0) return [0, 1];
  if (x < 0) x = -x;
  if (!tolerance) tolerance = 0.0001;
  var num = 1,
    den = 1;

  function iterate() {
    var R = num / den;
    if (Math.abs((R - x) / x) < tolerance) return;

    if (R < x) num++;
    else den++;
    iterate();
  }

  iterate();
  return [num, den];
}

createInputMatrix();
