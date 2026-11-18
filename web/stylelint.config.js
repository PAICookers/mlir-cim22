// @ts-check

export default {
  extends: ["stylelint-config-standard"],
  ignoreFiles: ["dist/**", "test-results/**"],
  rules: {
    "import-notation": "string",
    "at-rule-no-unknown": [true, { ignoreAtRules: ["theme"] }],
  },
};
