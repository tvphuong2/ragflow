import { isEmpty } from 'lodash';

class KeyGenerator {
  idx = 0;
  chars: string[] = [];
  constructor() {
    const chars = Array(26)
      .fill(1)
      .map((x, idx) => String.fromCharCode(97 + idx)); // 26 char
    this.chars = chars;
  }
  generateKey() {
    const key = this.chars[this.idx];
    this.idx++;
    return key;
  }
}

// Classify nodes based on edge relationships
export class Converter {
  keyGenerator;
  dict: Record<string, string> = {}; // key is node id, value is combo
  constructor() {
    this.keyGenerator = new KeyGenerator();
  }
  buildDict(edges: { source: string; target: string }[]) {
    edges.forEach((x) => {
      if (this.dict[x.source] && !this.dict[x.target]) {
        this.dict[x.target] = this.dict[x.source];
      } else if (!this.dict[x.source] && this.dict[x.target]) {
        this.dict[x.source] = this.dict[x.target];
      } else if (!this.dict[x.source] && !this.dict[x.target]) {
        this.dict[x.source] = this.dict[x.target] =
          this.keyGenerator.generateKey();
      }
    });
    return this.dict;
  }
  buildNodesAndCombos(nodes: any[], edges: any[]) {
    this.buildDict(edges);
    const nextNodes = nodes.map((x) => ({ ...x, combo: this.dict[x.id] }));

    const combos = Object.values(this.dict).reduce<any[]>((pre, cur) => {
      if (pre.every((x) => x.id !== cur)) {
        pre.push({
          id: cur,
          data: {
            label: `Combo ${cur}`,
          },
        });
      }
      return pre;
    }, []);

    return { nodes: nextNodes, combos };
  }
}

export const isDataExist = (data: any) => {
  return (
    data?.data && typeof data?.data !== 'boolean' && !isEmpty(data?.data?.graph)
  );
};

const normalizeComboId = (label: string) => {
  const cleaned = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

  return `combo-${cleaned || 'group'}`;
};

const findCombo = (communities: string[]) => {
  if (!Array.isArray(communities) || communities.length === 0) {
    return undefined;
  }

  const first = communities[0];
  if (typeof first === 'string') {
    return first;
  }

  return String(first ?? '').trim() || undefined;
};

const stripCombo = (node: any) => {
  if (!node || typeof node !== 'object') {
    return node;
  }

  const clone = { ...node } as Record<string, unknown>;
  if ('combo' in clone) {
    delete clone.combo;
  }

  return clone;
};

export interface GraphGroupResult {
  nodes: any[];
  combos: any[];
  hasCombos: boolean;
}

export const buildNodesAndCombos = (nodes: any[]): GraphGroupResult => {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return { nodes: [], combos: [], hasCombos: false };
  }

  const comboMap = new Map<string, { id: string; label: string }>();

  nodes.forEach((node) => {
    const comboLabel = findCombo(node?.communities);
    if (!comboLabel) {
      return;
    }

    if (!comboMap.has(comboLabel)) {
      comboMap.set(comboLabel, {
        id: normalizeComboId(comboLabel),
        label: comboLabel,
      });
    }
  });

  const combos = Array.from(comboMap.values()).map(({ id, label }) => ({
    isCombo: true,
    id,
    label,
    data: {
      label,
    },
  }));

  const nodesWithCombos = nodes.map((node) => {
    const comboLabel = findCombo(node?.communities);
    if (!comboLabel) {
      return stripCombo(node);
    }

    const combo = comboMap.get(comboLabel);
    if (!combo) {
      return stripCombo(node);
    }

    return {
      ...node,
      combo: combo.id,
    };
  });

  return {
    nodes: nodesWithCombos,
    combos,
    hasCombos: combos.length > 0,
  };
};
