package xfp.pdf.run;

import org.apache.pdfbox.pdmodel.PDDocument;
import xfp.pdf.arrange.MarkPdf;
import xfp.pdf.core.PdfParser;
import xfp.pdf.pojo.ContentPojo;
import xfp.pdf.tools.FileTool;

import java.io.File;
import java.io.IOException;


public class Pdf2html {

    public static void main(String[] args) throws IOException {

        File file = new File(Path.inputAllPdfPath);
        File[] files = file.listFiles();
        int id = 0;
        // 问题序列：184
        for (File f : files) {
            id += 1;
            if (id > 212) {
                System.out.println(id);
                PDDocument pdd = null;
                try {
                    pdd = PDDocument.load(f);
                    ContentPojo contentPojo = PdfParser.parsingUnTaggedPdfWithTableDetection(pdd);
                    MarkPdf.markTitleSep(contentPojo);
                    FileTool.saveHTML(Path.outputAllHtmlPath, contentPojo, f.getAbsolutePath());
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        pdd.close();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }
    }
}