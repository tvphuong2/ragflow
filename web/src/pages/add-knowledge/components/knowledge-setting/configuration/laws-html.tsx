import { Form } from 'antd';
import { useEffect } from 'react';

import { DocumentType } from '@/components/layout-recognize-form-field';

import { LawsConfiguration } from './laws';

export function LawsHtmlConfiguration() {
  const form = Form.useFormInstance();

  useEffect(() => {
    const current = form.getFieldValue(['parser_config', 'layout_recognize']);
    if (
      !current ||
      current === DocumentType.DeepDocVN ||
      current === DocumentType.DeepDOC
    ) {
      form.setFieldValue(
        ['parser_config', 'layout_recognize'],
        DocumentType.DeepDocHTML,
      );
    }
  }, [form]);

  return <LawsConfiguration />;
}
